"""DFST (Deep Feature Space Trojan) trigger generator.

Re-implementation of ``models.cyclegan.CycleGenerator`` from the original DFST code,
reconstructed layer-by-layer from ``cifar10_resnet18_dfst_generator.pt``. The generator is a
9-residual-block CycleGAN that maps a [0, 1] RGB image to its "sunrise"-styled version.
The state dict exported by ``convert_dfst.py`` loads into this class with strict=True.
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        self.norm = nn.InstanceNorm2d(dim)

    def forward(self, x):
        return x + self.norm(self.conv(x))


class CycleGenerator(nn.Module):
    def __init__(self, n_res=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3), nn.Conv2d(3, 32, 7), nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
        ]
        layers += [ResBlock(128) for _ in range(n_res)]
        layers += [
            nn.ConvTranspose2d(128, 256, 3, 1, 1), nn.PixelShuffle(2), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, 3, 1, 1), nn.PixelShuffle(2), nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3), nn.Conv2d(32, 3, 7), nn.Sigmoid(),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


def load_dfst_generator(path, device='cpu'):
    """Load a converted generator state dict (see convert_dfst.py) in eval mode."""
    g = CycleGenerator()
    g.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return g.to(device).eval()
