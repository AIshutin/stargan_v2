from models import StyleEncoder, MapperNetwork, Generator
from lpips_pytorch import LPIPS
from tqdm.auto import trange
import argparse
from pathlib import Path
import torch
import numpy as np
from train import to_image, transform
from datasets import CelebADataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--n_domains", type=int, default=10)    
    parser.add_argument("--attributes", nargs="+", default=["Blond_Hair", "Eyeglasses", "Goatee", 'Male', \
                                                            'Young', 'Black_Hair', 'Bald', 'Mouth_Slightly_Open', \
                                                            'Pale_Skin', 'Wearing_Necklace'])
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator()
    mapper = MapperNetwork(K_domains=args.n_domains)
    styler = StyleEncoder(args.n_domains)

    state = torch.load(args.checkpoint)
    generator.load_state_dict(state['generator_ema'])
    mapper.load_state_dict(state['mapper_ema'])
    styler.load_state_dict(state['styler_ema'])
    generator.to(device), mapper.to(device), styler.to(device)

    dataset = CelebADataset(transform=transform, attributes=args.attributes)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, # if bs > 1 then normalize lpips by bs
        num_workers=8,
        pin_memory=True)

    lpips = LPIPS()
    generator.eval()
    styler.eval()
    mapper.eval()
    
    test_iters = args.samples
    values = []
    for i in trange(test_iters):
        # fetch images and labels
        x_real, _ = next(iter(dataloader))
        x_ref, _ = next(iter(dataloader))
        x_ref2, _ = next(iter(dataloader))
        batch_size = x_real.shape[0]
        y_trg = torch.tensor(np.random.choice(np.arange(args.n_domains), size=batch_size)) # целевой домен
        y_org = torch.tensor(np.random.choice(np.arange(args.n_domains), size=batch_size)) # ваш домен
        
        x_real, x_ref, x_ref2 = [x.to(device).float() for x in [x_real, x_ref, x_ref2]]
        y_trg, y_org = [x.to(device).long() for x in [y_trg, y_org]]
        
        with torch.no_grad():
            x_fake = generator(x_real, mapper(torch.randn((y_trg.shape[0], 16), device=device), 
                                            y_trg))

        # HW statement requires me not to normalize these images in [-1, 1], but you may wish to do so
        values.append(lpips(x_fake.cpu(), x_real.cpu()).squeeze().item())

    print('lpips', np.mean(values))
    assert(np.mean(values) < 1.3)