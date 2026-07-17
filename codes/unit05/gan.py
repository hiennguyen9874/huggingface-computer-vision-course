"""A compact DCGAN and its two adversarial objectives.

Run: uv run python codes/unit05/gan.py
N=batch, Z=noise width. Images use float32 values in `[-1, 1]`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Generator(nn.Module):
    """Generate RGB images `[N, 3, 32, 32]` from Gaussian noise `[N, Z]`."""

    def __init__(self, noise_dim: int = 64) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.project = nn.Linear(noise_dim, 128 * 4 * 4)
        self.blocks = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # [N,128,4,4] -> [N,64,8,8]
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # -> [N,32,16,16]
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),  # -> [N,3,32,32]
        )

    def forward_with_shapes(self, noise: Tensor) -> dict[str, Tensor]:
        if noise.ndim != 2 or noise.shape[1] != self.noise_dim:
            raise ValueError(f"expected noise [N, {self.noise_dim}], got {tuple(noise.shape)}")
        projected = self.project(noise)  # [N, Z] -> [N, 128*4*4]
        feature_map = projected.reshape(noise.shape[0], 128, 4, 4)
        image = self.blocks(feature_map)  # [N, 3, 32, 32]
        return {"noise": noise, "projected": projected, "feature_map": feature_map, "fake_image": image}

    def forward(self, noise: Tensor) -> Tensor:
        return self.forward_with_shapes(noise)["fake_image"]


class Discriminator(nn.Module):
    """Map RGB images `[N, 3, 32, 32]` to real/fake logits `[N]`."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2),       # -> [N,32,16,16]
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),  # -> [N,64,8,8]
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), # -> [N,128,4,4]
        )
        self.classifier = nn.Linear(128 * 4 * 4, 1)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
            raise ValueError(f"expected images [N, 3, 32, 32], got {tuple(images.shape)}")
        features = self.features(images)
        logits = self.classifier(features.flatten(1)).squeeze(1)  # [N,1] -> [N]
        return {"images": images, "features": features, "real_fake_logits": logits}

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["real_fake_logits"]


def gan_losses(discriminator: Discriminator, real: Tensor, fake: Tensor) -> tuple[Tensor, Tensor]:
    """Return `(discriminator_loss, generator_loss)` scalar BCE objectives.

    Fake images are detached only for the discriminator objective. The generator
    uses the non-saturating target "make D(fake) predict real" for stronger gradients.
    """
    real_logits, detached_fake_logits = discriminator(real), discriminator(fake.detach())
    discriminator_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    discriminator_loss += F.binary_cross_entropy_with_logits(detached_fake_logits, torch.zeros_like(detached_fake_logits))
    generator_logits = discriminator(fake)
    generator_loss = F.binary_cross_entropy_with_logits(generator_logits, torch.ones_like(generator_logits))
    return discriminator_loss, generator_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    generator, discriminator = Generator().eval(), Discriminator().eval()
    noise, real = torch.randn(2, 64), torch.randn(2, 3, 32, 32).tanh()
    with torch.no_grad():
        generated_steps = generator.forward_with_shapes(noise)
        judged_steps = discriminator.forward_with_shapes(generated_steps["fake_image"])
        d_loss, g_loss = gan_losses(discriminator, real, generated_steps["fake_image"])
    for name, value in {**generated_steps, **judged_steps}.items():
        print(f"{name:18} shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"discriminator_loss={d_loss:.4f}, generator_loss={g_loss:.4f}")
