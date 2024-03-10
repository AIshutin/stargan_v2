from models import Generator, Discriminator, MapperNetwork, StyleEncoder, \
                    he_initialization_lrelu, he_initialization_relu
from datasets import CelebADataset, BiDataset
import torch
from torchvision import transforms
import argparse
from tqdm.auto import tqdm
import wandb
from utils import *
import os
import copy


def calculate_generator_loss(styles1, styles2, images, nets, criterions, cyc_weight, div_weight, noise_lev=0.0):
    images_hat = nets.generator(images, styles1)
    styles_true = nets.styler(images, true_domains)
    t2 = discriminator(images_hat + torch.randn_like(images_hat) * noise_lev, domains)

    loss_cyc = criterions.cyc(nets.generator(images_hat, styles_true), images)
    loss_adv = criterions.adv(t2, torch.ones_like(t2))
    loss_sty = criterions.rec(nets.styler(images_hat, domains), styles1)
    loss_div = -criterions.div(images_hat, nets.generator(images, styles2).detach())
    loss     = cyc_weight * loss_cyc + loss_adv + loss_sty + div_weight * loss_div

    return loss, dict(adv=loss_adv.item(),
                      sty=loss_sty.item(),
                      div=loss_div.item(),
                      cyc=loss_cyc.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--n_val_images", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise_iters", type=int, default=20000)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--gp_weight", type=float, default=1)
    parser.add_argument("--cyc_weight", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--attributes", nargs="+", default=["Blond_Hair", "Eyeglasses", "No_Beard"])
    parser.add_argument("--ema", type=float, default=0.999)
    args = parser.parse_args()
    # Spatial size of training images, images are resized to this size.
    args.img_size = 256 

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    transform=transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means,
                            std=stds)
    ])

    to_image=transforms.Compose([
        transforms.Normalize(std=[1 / el for el in stds], mean=[0] * 3),
        transforms.Normalize(std=[1] * 3, mean=[-el for el in means]),
        transforms.ToPILImage()
    ])

    # Load the dataset from file and apply transformations
    dataset = CelebADataset(transform=transform, attributes=args.attributes)
    bidataset = BiDataset(dataset)

    args.num_domains = len(dataset.header)

    # Number of workers for the dataloader
    num_workers = 0 if device.type == 'cuda' else 2
    # Whether to put fetched data tensors to pinned memory
    pin_memory = True if device.type == 'cuda' else False

    # dataloader for batched data loading
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory,
                                            sampler=torch.utils.data.RandomSampler(dataset, 
                                                                                   replacement=True, 
                                                                                   num_samples=args.batch_size * args.iterations))
    bidataloader = torch.utils.data.DataLoader(bidataset,
                                            batch_size=args.batch_size,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory,
                                            sampler=torch.utils.data.RandomSampler(bidataset, 
                                                                                   replacement=True, 
                                                                                   num_samples=args.batch_size * args.iterations))

    generator = Generator()
    mapper = MapperNetwork(K_domains=args.num_domains)
    styler = StyleEncoder(K_domains=args.num_domains)
    discriminator = Discriminator(K_domains=args.num_domains)

    nets = SuperDict(generator=nn.DataParallel(generator),
                     mapper=nn.DataParallel(mapper),
                     styler=nn.DataParallel(styler),
                     discriminator=nn.DataParallel(discriminator))
    nets.apply(he_initialization_relu)
    nets_ema = SuperDict(generator=copy.deepcopy(generator),
                         mapper=copy.deepcopy(mapper),
                         styler=copy.deepcopy(styler))
    
    nets.applym('to', device)
    nets_ema.applym('to', device)

    optims = SuperDict(
        s=torch.optim.Adam(nets.styler.parameters(), 
                           betas=(0, 0.99), lr=1e-4, weight_decay=args.weight_decay),
        g=torch.optim.Adam(nets.generator.parameters(), 
                           betas=(0, 0.99), lr=1e-4, weight_decay=args.weight_decay),
        f=torch.optim.Adam(nets.mapper.parameters(), 
                           betas=(0, 0.99), lr=1e-6, weight_decay=args.weight_decay),
        d=torch.optim.Adam(nets.discriminator.parameters(), 
                           betas=(0, 0.99), lr=1e-4 * 2, weight_decay=args.weight_decay)
    )
    tracker = MetricTracker()

    criterions = SuperDict(
        adv=torch.nn.BCEWithLogitsLoss(),
        div=torch.nn.L1Loss(),
        rec=torch.nn.L1Loss(),
        cyc=torch.nn.L1Loss()
    )

    wandb.init(
        project="stargan2",
        config=args
    )
    wandb.run.log_code(".")

    for n_iter, ((images, meta), (images1, images2, domains)) in enumerate((zip(tqdm(dataloader), bidataloader))):
            if n_iter > args.iterations:
                break
            images = images.to(device)
            images.requires_grad = True
            images1 = images1.to(device)
            images2 = images2.to(device)
            domains = domains.to(device)

            attributes = meta['attributes'].to(device)
            if args.noise_iters < n_iter:
                noise_lev = 0
            else:
                noise_lev = args.noise * (1 - n_iter / args.noise_iters)
            true_domains = get_true_domains(attributes)

            # DISCRIMINATOR TRAINING
            optims.d.zero_grad()
            domains = torch.randint(high=args.num_domains, size=(images.shape[0],))
            noise = torch.randn(images.shape[0], 16, device=device)
            with torch.no_grad():
                mapper_styles = nets.mapper(noise, domains)
                styler_styles = nets.styler(images1, domains)
                images_hat    = nets.generator(torch.cat((images, images)), 
                                                torch.cat((mapper_styles, styler_styles)))

            # Adversarial loss
            t1 = nets.discriminator(images + torch.randn_like(images) * noise_lev, true_domains)
            t2 = nets.discriminator(images_hat + torch.randn_like(images_hat) * noise_lev, 
                                    torch.cat((domains, domains)))
            loss_adv = criterions.adv(t1, apply_label_smoothing(torch.ones_like(t1), args.label_smoothing)) \
                     + criterions.adv(t2, apply_label_smoothing(torch.zeros_like(t2), args.label_smoothing))

            # R1 grad penalty from https://arxiv.org/pdf/1801.04406.pdf
            grads = torch.autograd.grad(
                inputs=images,
                outputs=t1.sum(), # mean?
                retain_graph=True, create_graph=True, only_inputs=True)[0]
            loss_reg = 0.5 * grads.square().sum() / grads.shape[0]
            
            loss = loss_reg * args.gp_weight + loss_adv
            loss.backward()
            tracker.add('d_loss_adv_d', loss_adv.item())
            tracker.add('d_loss_reg_d', loss_reg.item())
            tracker.add('d_loss', loss.item())
            tracker.add('d_grad_norm', calc_grad_norm(nets.discriminator.parameters()))
            optims.d.step()

            # GENERATOR TRAINING
            # Note that if done in one pass, the training will require more than 24 GB VRAM
            # Also, an interpolation is not fully supported in halfprecision in torch 2.0 
            # Thus, we will do it in 2 passes
            optims.d.zero_grad()
            optims.f.zero_grad()
            optims.s.zero_grad()
            optims.g.zero_grad()
            # MAPPER pass
            style1 = nets.mapper(torch.randn(images.shape[0], 16, device=device), domains)
            style2 = nets.mapper(torch.randn(images.shape[0], 16, device=device), domains)
            g_loss, loss_dict = calculate_generator_loss(
                style1, style2, images, nets, criterions, args.cyc_weight, 
                1 - n_iter / args.iterations, noise_lev
            )
            g_loss.backward()
            optims.g.step()
            optims.f.step()
            optims.s.step()

            for key, value in loss_dict.items():
                tracker.add('g1_' + key, value)
            tracker.add('g1_grad_norm', calc_grad_norm(list(nets.generator.parameters()) +\
                                                       list(nets.mapper.parameters()) +\
                                                       list(nets.styler.parameters())))
            
            optims.d.zero_grad()
            optims.f.zero_grad()
            optims.s.zero_grad()
            optims.g.zero_grad()
            # STYLER pass
            style1 = nets.styler(images1, domains)
            style2 = nets.styler(images2, domains)
            loss, loss_dict = calculate_generator_loss(
                style1, style2, images, nets, criterions, args.cyc_weight, 
                1 - n_iter / args.iterations, noise_lev
            )
            loss.backward()
            optims.g.step()
            
            for key, value in loss_dict.items():
                tracker.add('g2_' + key, value)
            tracker.add('g2_grad_norm', calc_grad_norm(nets.generator.parameters()))

            ema(nets_ema.generator, nets.generator.module, beta=args.ema)
            ema(nets_ema.mapper, nets.mapper.module, beta=args.ema)
            ema(nets_ema.styler, nets.styler.module, beta=args.ema)

            if (n_iter + 1) % args.log_steps == 0:
                with torch.inference_mode():
                    val_images = []
                    styles = nets_ema.styler(images[0].unsqueeze(0).repeat(args.num_domains, 1, 1, 1), 
                                                    torch.arange(args.num_domains, device=device))
                    for i in range(args.n_val_images):
                        val_images.append([wandb.Image(to_image(images[i]))])
                        x = images[i].unsqueeze(0)
                        x = torch.cat([x.clone() for j in range(args.num_domains)], dim=0)
                        new_images = nets_ema.generator(x, styles)
                        for j in range(args.num_domains):
                            val_images[-1].append(wandb.Image(to_image(new_images[j])))

                    table = wandb.Table(data=val_images, columns=['original'] + dataset.header)

                    wandb.log({
                        **tracker.to_dict(),
                        "images": table
                    })
                    tracker.reset()

            
            if (n_iter + 1) % args.save_steps == 0:
                os.system('mkdir checkpoints')     
                os.system(f"mkdir checkpoints/{wandb.run.id}")       
                states = {
                    "generator": nets.generator.state_dict(),
                    "generator_ema": nets_ema.generator.state_dict(),
                    "discriminator": nets.discriminator.state_dict(),
                    "styler": nets.styler.state_dict(),
                    "styler_ema": nets_ema.styler.state_dict(),
                    "mapper": nets.mapper.state_dict(),
                    "mapper_ema": nets_ema.mapper.state_dict(),
                    "optim_g": optims.g.state_dict(),
                    "optim_d": optims.d.state_dict(),
                    "optim_f": optims.f.state_dict(),
                    "optim_s": optims.s.state_dict(),
                }
                torch.save(states, f"checkpoints/{wandb.run.id}/checkpoint-{n_iter}.pt")
