"""Offline miniature of Stable Diffusion's VAE + text + U-Net data flow.

It demonstrates contracts and cross-attention, but has random weights and is not
a replacement for a pretrained Hugging Face Diffusers pipeline.
Run: uv run python codes/unit05/stable_diffusion.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LatentAutoencoder(nn.Module):
    """Compress `[N,3,32,32]` pixels to `[N,4,8,8]` and decode them back."""

    def __init__(self, latent_channels: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),  # -> [N,32,16,16]
            nn.Conv2d(32, latent_channels, 4, 2, 1), # -> [N,4,8,8]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )

    def encode(self, images: Tensor) -> Tensor:
        return self.encoder(images)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)


class TinyTextEncoder(nn.Module):
    """Encode token IDs `[N,L]` into contextual prompt embeddings `[N,L,D]`."""

    def __init__(self, vocabulary_size: int = 1000, width: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 1)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.transformer(self.embedding(token_ids))


class CrossAttentionDenoiser(nn.Module):
    """Predict latent noise `[N,4,8,8]`, conditioned on text `[N,L,D]` and `t`.

    Spatial latent positions are queries; prompt tokens provide keys and values.
    This is the central text-image fusion mechanism used throughout SD's U-Net.
    """

    def __init__(self, latent_channels: int = 4, width: int = 32) -> None:
        super().__init__()
        self.input = nn.Conv2d(latent_channels, width, 3, padding=1)
        self.time_embedding = nn.Embedding(1000, width)
        self.cross_attention = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.feed_forward = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.output = nn.Conv2d(width, latent_channels, 3, padding=1)

    def forward_with_shapes(self, noisy_latents: Tensor, timesteps: Tensor, text: Tensor) -> dict[str, Tensor]:
        spatial = self.input(noisy_latents)  # [N,4,H,W] -> [N,D,H,W]
        batch, width, height, image_width = spatial.shape
        queries = spatial.flatten(2).transpose(1, 2)  # [N,D,H,W] -> [N,H*W,D]
        timed_queries = queries + self.time_embedding(timesteps)[:, None, :]  # [N,H*W,D]
        attended, attention_weights = self.cross_attention(timed_queries, text, text)
        tokens = timed_queries + attended
        tokens = tokens + self.feed_forward(tokens)
        feature_map = tokens.transpose(1, 2).reshape(batch, width, height, image_width)
        predicted_noise = self.output(feature_map)  # [N,D,H,W] -> [N,4,H,W]
        return {"noisy_latents": noisy_latents, "spatial_queries": queries,
                "text_embeddings": text, "attention_weights": attention_weights,
                "conditioned_features": feature_map, "predicted_noise": predicted_noise}

    def forward(self, noisy_latents: Tensor, timesteps: Tensor, text: Tensor) -> Tensor:
        return self.forward_with_shapes(noisy_latents, timesteps, text)["predicted_noise"]


class TinyStableDiffusion(nn.Module):
    """Compose the three SD modules and expose a complete training-step trace."""

    def __init__(self) -> None:
        super().__init__()
        self.vae = LatentAutoencoder()
        self.text_encoder = TinyTextEncoder()
        self.denoiser = CrossAttentionDenoiser()

    def forward_with_shapes(self, images: Tensor, token_ids: Tensor, timesteps: Tensor) -> dict[str, Tensor]:
        clean_latents = self.vae.encode(images)  # pixel space -> cheaper latent space
        noise = torch.randn_like(clean_latents)
        # A simplified alpha_bar schedule; `diffusion.py` implements the exact
        # cumulative DDPM schedule. Squared coefficients sum to one.
        noise_fraction = ((timesteps.float() + 1) / 1000)[:, None, None, None]
        noisy_latents = (1 - noise_fraction).sqrt() * clean_latents + noise_fraction.sqrt() * noise
        text = self.text_encoder(token_ids)
        denoising = self.denoiser.forward_with_shapes(noisy_latents, timesteps, text)
        decoded = self.vae.decode(clean_latents)
        return {"images": images, "clean_latents": clean_latents, "sampled_noise": noise,
                **denoising, "decoded_images": decoded}


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyStableDiffusion().eval()
    images = torch.randn(2, 3, 32, 32).tanh()
    token_ids = torch.randint(0, 1000, (2, 12))  # SD uses up to 77 tokens; 12 keeps the demo small
    timesteps = torch.tensor([100, 800])
    with torch.no_grad():
        steps = model.forward_with_shapes(images, token_ids, timesteps)
    print(f"token_ids            shape={tuple(token_ids.shape)}, dtype={token_ids.dtype}")
    for name, value in steps.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
