import torch
from torch import nn
from torch.nn import functional as F
from utils import *

RELU_SLOPE = 0.1
normalize_layer = lambda x: x # nn.utils.spectral_norm #


def he_initialization_lrelu(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight, RELU_SLOPE, nonlinearity='leaky_relu') # or relu and 0 slope
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    else:
        for subm in m.children():
            subm.apply(he_initialization_lrelu)


def he_initialization_relu(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu') # or relu and 0 slope
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    else:
        for subm in m.children():
            subm.apply(he_initialization_relu)


class UpsamplingResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super().__init__()
        self.norm1 = AdaIN(in_channels)
        self.norm2 = AdaIN(out_channels)
        self.conv1 = normalize_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = normalize_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.act = nn.LeakyReLU(RELU_SLOPE)
        self.upsample = F.interpolate
    
    def forward(self, X, style):
        X = self.norm1(X, style)
        X = self.act(X)
        X = self.upsample(X, scale_factor=(2, 2), mode='nearest-exact')
        X = self.conv1(X)

        X = self.norm2(X, style)
        X = self.act(X)
        X = self.conv2(X)
        return X


class AdaIN(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()
        self.mu = normalize_layer(nn.Linear(64, num_features))
        self.std = normalize_layer(nn.Linear(64, num_features))
        self.IN = nn.InstanceNorm2d(num_features=num_features, affine=False)

    def forward(self, X, style):
        X_norm = self.IN(X)
        X_norm = X_norm * (1 + self.std(style).reshape(X.shape[0], -1, 1, 1)) + self.mu(style).reshape(X.shape[0], -1, 1, 1)
        return X_norm


class Generator(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.prenet = nn.Sequential(
            normalize_layer(nn.Conv2d( 3, 64, kernel_size=3, padding=1)),
            ResBlock( 64, 128, 'avg', 'in'),
            ResBlock(128, 256, 'avg', 'in'),
            ResBlock(256, 512, 'avg', 'in'),
            ResBlock(512, 512, 'avg', 'in'),
            ResBlock(512, 512, 'none', 'in'),
            ResBlock(512, 512, 'none', 'in'),
        )

        self.net = nn.ModuleList(
            [
                ResBlock(512, 512, 'none', 'adain'),
                ResBlock(512, 512, 'none', 'adain'),
                UpsamplingResBlock(512, 512, ),
                UpsamplingResBlock(512, 256, ),
                UpsamplingResBlock(256, 128, ),
                UpsamplingResBlock(128, 64, )
            ]
        )
        self.post = nn.Sequential(
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(RELU_SLOPE),
            normalize_layer(nn.Conv2d(64, 3, kernel_size=3, padding=1)),
        )

    def forward(self, X, style):
        X = self.prenet(X)
        for layer in self.net:
            X = layer(X, style)
        return self.post(X)


class MapperNetwork(nn.Module):
    def __init__(self, input_dim=16, middle_dim=512, output_dim=64, K_domains=1):
        super().__init__()
        self.shared = nn.Sequential(
            normalize_layer(nn.Linear(input_dim, middle_dim)),
            nn.ReLU(),
            normalize_layer(nn.Linear(middle_dim, middle_dim)),
            nn.ReLU(),
            normalize_layer(nn.Linear(middle_dim, middle_dim)),
            nn.ReLU(),
            normalize_layer(nn.Linear(middle_dim, middle_dim)),
            nn.ReLU(),
        )
        self.branches = nn.ModuleList()
        for i in range(K_domains):
            self.branches.append(
                nn.Sequential(
                    normalize_layer(nn.Linear(middle_dim, middle_dim)),
                    nn.ReLU(),
                    normalize_layer(nn.Linear(middle_dim, middle_dim)),
                    nn.ReLU(),
                    normalize_layer(nn.Linear(middle_dim, middle_dim)),
                    nn.ReLU(),
                    normalize_layer(nn.Linear(middle_dim, output_dim))
                )
            )

    def forward(self, X, domains):
        X = self.shared(X)
        branch_specific = []
        for i, el in enumerate(domains):
            branch_specific.append(self.branches[el](X[i].unsqueeze(0)))
        output = torch.cat(branch_specific, dim=0)
        assert(output.shape[0] == X.shape[0])
        assert(len(output.shape) == 2)
        return output


class ResBlock(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, resampling='none', norm='none'):
        super().__init__()
        self.conv1 = normalize_layer(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = normalize_layer(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.proj = normalize_layer(nn.Conv2d(in_channels, out_channels, 1, bias=False)) \
                    if out_channels != in_channels else lambda x: x
        self.proj_w = 1.0 #nn.Parameter(torch.tensor([1.0]))
        self.act = nn.LeakyReLU(RELU_SLOPE)
        self.resampling = lambda x: x
        self.norm1 = lambda x: x
        self.norm2 = lambda x: x

        if resampling == 'avg':
            self.resampling = nn.AvgPool2d(kernel_size=2)
        if norm == 'in':
            self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'adain':
            self.norm1 = AdaIN(in_channels)
            self.norm2 = AdaIN(out_channels)

    def forward(self, X, style=None):
        X_proj = self.resampling(self.proj(X))
        
        if style is not None:
            X = self.norm1(X, style)
        else:
            X = self.norm1(X)
        X = self.act(X)
        X = self.conv1(X)
        X = self.resampling(X)
        if style is None:
            X = self.norm2(X)
        else:
            X = self.norm2(X, style)
        X = self.act(X)
        X = self.conv2(X)
        return (X_proj * self.proj_w + X) / (1 + self.proj_w) ** 0.5 # adaptive residual



class Reshape(nn.Module):
    def __init__(self, *shape) -> None:
        super().__init__()
        self.shape = [-1] + list(shape)
    
    def forward(self, X):
        return X.reshape(self.shape)


class StyleEncoder(nn.Module):
    def __init__(self, K_domains=1, D=64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            normalize_layer(nn.Conv2d(3, 64, kernel_size=1)),
            ResBlock(64,  128, 'avg'),
            ResBlock(128, 256, 'avg'),
            ResBlock(256, 512, 'avg'),
            ResBlock(512, 512, 'avg'),
            ResBlock(512, 512, 'avg'),
            ResBlock(512, 512, 'avg'),
            # Maybe blocks below are domain-specific
            nn.LeakyReLU(RELU_SLOPE),
            nn.Conv2d(512, 512, kernel_size=4),
            nn.LeakyReLU(RELU_SLOPE),
            Reshape(512),
        )
        self.branches = nn.ModuleList()
        for i in range(K_domains):
            self.branches.append(
                nn.Sequential(
                    nn.Linear(512, D)
                )
            )
    
    def forward(self, X, domains):
        X = self.shared(X)
        branch_specific = []
        for i, el in enumerate(domains):
            branch_specific.append(self.branches[el](X[i].unsqueeze(0)))
        output = torch.cat(branch_specific, dim=0)
        assert(output.shape[0] == X.shape[0])
        assert(len(output.shape) == 2)
        return output


class Discriminator(StyleEncoder):
    def __init__(self, K_domains=1) -> None:
        super().__init__(K_domains=K_domains, D=1)
