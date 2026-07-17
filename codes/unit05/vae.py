"""Convolutional Variational Autoencoder (VAE).

Run: uv run python codes/unit05/vae.py
The file includes deterministic encoding, reparameterized sampling, decoding,
and the reconstruction + KL objective described in Unit 5.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VariationalAutoencoder(nn.Module):
    """Map `[N, 1, 28, 28]` images through a Gaussian latent `[N, Z]`.

    Inputs are float images in `[0, 1]`. The decoder returns probabilities in
    `[0, 1]` with the same shape. During evaluation, pass `sample=False` for a
    deterministic reconstruction using `z=mu`.
    """

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),  # [N,1,28,28] -> [N,32,14,14]
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), # -> [N,64,7,7]
            nn.Flatten(),
        )
        self.to_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.to_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # -> [N,32,14,14]
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(), # -> [N,1,28,28]
        )

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError(f"expected images [N, 1, 28, 28], got {tuple(images.shape)}")
        features = self.encoder(images)  # [N, 64*7*7]
        return self.to_mu(features), self.to_logvar(features)  # each [N, Z]

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample `z = mu + exp(0.5*logvar)*epsilon` without blocking gradients."""
        standard_deviation = torch.exp(0.5 * logvar)  # [N, Z]
        epsilon = torch.randn_like(standard_deviation)  # external randomness [N, Z]
        return mu + standard_deviation * epsilon

    def decode(self, latent: Tensor) -> Tensor:
        features = self.from_latent(latent).reshape(latent.shape[0], 64, 7, 7)
        return self.decoder(features)  # [N, Z] -> [N, 1, 28, 28]

    def forward_with_shapes(self, images: Tensor, sample: bool = True) -> dict[str, Tensor]:
        features = self.encoder(images)  # shown separately for an executable shape trace
        mu, logvar = self.to_mu(features), self.to_logvar(features)
        latent = self.reparameterize(mu, logvar) if sample else mu
        decoder_input = self.from_latent(latent).reshape(images.shape[0], 64, 7, 7)
        reconstruction = self.decoder(decoder_input)
        return {"images": images, "encoder_features": features, "mu": mu, "logvar": logvar,
                "latent": latent, "decoder_features": decoder_input, "reconstruction": reconstruction}

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        steps = self.forward_with_shapes(images)
        return steps["reconstruction"], steps["mu"], steps["logvar"]


def vae_loss(reconstruction: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1.0) -> dict[str, Tensor]:
    """Return scalar mean reconstruction, KL, and beta-weighted total losses."""
    reconstruction_loss = F.binary_cross_entropy(reconstruction, target, reduction="sum") / target.shape[0]
    kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / target.shape[0]
    return {"reconstruction_loss": reconstruction_loss, "kl_loss": kl_loss,
            "total_loss": reconstruction_loss + beta * kl_loss}


if __name__ == "__main__":
    torch.manual_seed(0)
    model = VariationalAutoencoder(latent_dim=16).eval()
    images = torch.rand(2, 1, 28, 28)
    with torch.no_grad():
        steps = model.forward_with_shapes(images)
        losses = vae_loss(steps["reconstruction"], images, steps["mu"], steps["logvar"])
        samples = model.decode(torch.randn(2, 16))  # prior z~N(0,I) creates new images
    for name, value in steps.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"prior_samples        shape={tuple(samples.shape)}")
    print(", ".join(f"{name}={value:.4f}" for name, value in losses.items()))
