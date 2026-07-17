"""Tiny StyleGAN1-style synthesis: mapping, AdaIN, and noise injection.

This is an architectural lesson, not NVIDIA's production StyleGAN checkpoint.
Run: uv run python codes/unit05/stylegan.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MappingNetwork(nn.Module):
    """Map entangled Gaussian codes `[N, Z]` to intermediate styles `[N, W]`."""

    def __init__(self, latent_dim: int = 64, style_dim: int = 64, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(depth):
            layers.extend((nn.Linear(latent_dim if index == 0 else style_dim, style_dim), nn.LeakyReLU(0.2)))
        self.layers = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        # Normalizing z prevents its magnitude from becoming an accidental style signal.
        return self.layers(F.normalize(latent, dim=-1))  # [N, Z] -> [N, W]


class AdaIN(nn.Module):
    """Normalize features `[N,C,H,W]`, then apply scale/bias derived from `[N,W]`."""

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.affine = nn.Linear(style_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        with torch.no_grad():
            self.affine.bias[:channels].fill_(1)  # initial scale=1; bias=0

    def forward(self, features: Tensor, style: Tensor) -> Tensor:
        scale, bias = self.affine(style).chunk(2, dim=1)  # each [N, C]
        normalized = F.instance_norm(features)  # per-sample, per-channel normalization
        return scale[:, :, None, None] * normalized + bias[:, :, None, None]


class StyledBlock(nn.Module):
    """Upsample, convolve, inject pixel noise, and apply one style."""

    def __init__(self, in_channels: int, out_channels: int, style_dim: int, upsample: bool = True) -> None:
        super().__init__()
        self.upsample = upsample
        self.convolution = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.noise_strength = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.1)
        self.adain = AdaIN(out_channels, style_dim)

    def forward(self, features: Tensor, style: Tensor, noise: Tensor | None = None) -> Tensor:
        if self.upsample:
            features = F.interpolate(features, scale_factor=2, mode="nearest")
        features = self.convolution(features)
        if noise is None:
            noise = torch.randn(features.shape[0], 1, *features.shape[-2:], device=features.device)
        features = features + self.noise_strength * noise  # shared map, learned strength per channel
        return F.leaky_relu(self.adain(features, style), 0.2)


class TinyStyleGenerator(nn.Module):
    """Generate `[N,3,32,32]` images from `[N,64]` latent codes.

    The learned 4x4 constant provides spatial input. The same `w` controls every
    resolution; production StyleGAN also supports style mixing with different w's.
    """

    def __init__(self, latent_dim: int = 64, style_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.mapping = MappingNetwork(latent_dim, style_dim)
        self.constant = nn.Parameter(torch.randn(1, 64, 4, 4))
        self.blocks = nn.ModuleList((
            StyledBlock(64, 64, style_dim, upsample=False), # 4x4: coarse pose/shape
            StyledBlock(64, 64, style_dim),                 # 8x8
            StyledBlock(64, 32, style_dim),                 # 16x16
            StyledBlock(32, 16, style_dim),                 # 32x32: fine texture
        ))
        self.to_rgb = nn.Conv2d(16, 3, 1)

    def forward_with_shapes(self, latent: Tensor) -> dict[str, Tensor]:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"expected latent [N, {self.latent_dim}], got {tuple(latent.shape)}")
        style = self.mapping(latent)
        features = self.constant.expand(latent.shape[0], -1, -1, -1)
        trace = {"latent_z": latent, "intermediate_w": style, "constant": features}
        for index, block in enumerate(self.blocks):
            features = block(features, style)
            trace[f"styled_block_{index}"] = features
        trace["image"] = self.to_rgb(features).tanh()
        return trace

    def forward(self, latent: Tensor) -> Tensor:
        return self.forward_with_shapes(latent)["image"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyStyleGenerator().eval()
    with torch.no_grad():
        steps = model.forward_with_shapes(torch.randn(2, 64))
    for name, value in steps.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
