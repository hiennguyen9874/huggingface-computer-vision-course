"""CycleGAN for unpaired image-to-image translation.

Contains two reusable architecture types (instantiate G/F and D_X/D_Y) and all
three Unit 5 objectives: least-squares adversarial, cycle, and identity losses.
Run: uv run python codes/unit05/cyclegan.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Preserve `[N,C,H,W]` while learning a residual image transformation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels), nn.ReLU(),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.block(inputs)


class CycleGenerator(nn.Module):
    """Translate RGB `[N,3,32,32]` between domains, preserving the same shape."""

    def __init__(self, residual_blocks: int = 2) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(3, 32, 7), nn.InstanceNorm2d(32), nn.ReLU(), # [N,32,32,32]
            nn.Conv2d(32, 64, 3, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(), # [N,64,16,16]
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(), # [N,128,8,8]
        )
        self.residuals = nn.Sequential(*(ResidualBlock(128) for _ in range(residual_blocks)))
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.InstanceNorm2d(32), nn.ReLU(),
            nn.ReflectionPad2d(3), nn.Conv2d(32, 3, 7), nn.Tanh(),
        )

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        downsampled = self.downsample(images)
        transformed = self.residuals(downsampled)
        translated = self.upsample(transformed)
        return {"input_images": images, "downsampled": downsampled,
                "residual_features": transformed, "translated_images": translated}

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["translated_images"]


class PatchDiscriminator(nn.Module):
    """Classify overlapping patches: `[N,3,32,32] -> [N,1,2,2]` logits."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2), # -> 16x16
            nn.Conv2d(32, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2), # -> 8x8
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2), # -> 4x4
            nn.Conv2d(128, 1, 3, 1, 0), # -> 2x2 local decisions
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.layers(images)


def least_squares_discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    """Scalar LSGAN discriminator loss: real patches -> 1, fake patches -> 0."""
    return 0.5 * ((real_logits - 1).square().mean() + fake_logits.square().mean())


def cycle_generator_losses(
    fake_x_logits: Tensor, fake_y_logits: Tensor, real_x: Tensor, real_y: Tensor,
    cycled_x: Tensor, cycled_y: Tensor, identity_x: Tensor, identity_y: Tensor,
    lambda_cycle: float = 10.0, lambda_identity: float = 5.0,
) -> dict[str, Tensor]:
    """Return scalar adversarial, cycle L1, identity L1, and total G/F losses."""
    adversarial = 0.5 * ((fake_x_logits - 1).square().mean() + (fake_y_logits - 1).square().mean())
    cycle = F.l1_loss(cycled_x, real_x) + F.l1_loss(cycled_y, real_y)
    identity = F.l1_loss(identity_x, real_x) + F.l1_loss(identity_y, real_y)
    return {"generator_adversarial": adversarial, "cycle_consistency": cycle,
            "identity": identity, "generator_total": adversarial + lambda_cycle * cycle + lambda_identity * identity}


if __name__ == "__main__":
    torch.manual_seed(0)
    g_x_to_y, f_y_to_x = CycleGenerator().eval(), CycleGenerator().eval()
    d_x, d_y = PatchDiscriminator().eval(), PatchDiscriminator().eval()
    real_x, real_y = torch.randn(2, 3, 32, 32).tanh(), torch.randn(2, 3, 32, 32).tanh()
    with torch.no_grad():
        x_to_y = g_x_to_y.forward_with_shapes(real_x)
        fake_y = x_to_y["translated_images"]
        fake_x = f_y_to_x(real_y)
        cycled_x, cycled_y = f_y_to_x(fake_y), g_x_to_y(fake_x)
        identity_x, identity_y = f_y_to_x(real_x), g_x_to_y(real_y)
        logits = {"D_X(real_x)": d_x(real_x), "D_X(fake_x)": d_x(fake_x),
                  "D_Y(real_y)": d_y(real_y), "D_Y(fake_y)": d_y(fake_y)}
        losses = cycle_generator_losses(logits["D_X(fake_x)"], logits["D_Y(fake_y)"], real_x, real_y,
                                        cycled_x, cycled_y, identity_x, identity_y)
    for name, value in x_to_y.items():
        print(f"G X->Y {name:16} shape={tuple(value.shape)}")
    for name, value in logits.items():
        print(f"{name:24} shape={tuple(value.shape)}")
    print(f"cycled_x={tuple(cycled_x.shape)}, cycled_y={tuple(cycled_y.shape)}")
    print(", ".join(f"{name}={value:.4f}" for name, value in losses.items()))
