"""Autoregressive and latent-diffusion approaches to text-to-image generation.

Run with:
    uv run python codes/unit04/text_to_image.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AutoregressiveImageTokenModel(nn.Module):
    """Predict discrete image tokens conditioned on text.

    Inputs:
        text_ids: integer tokens `[batch, text_length]`.
        image_token_ids: shifted-right image tokens `[batch, image_length]`.
    Output:
        logits `[batch, image_length, image_vocabulary_size]`.

    A VQ-VAE would decode predicted image-token IDs into pixels. Causal masking
    ensures position `t` cannot inspect future image tokens.
    """

    def __init__(
        self, text_vocabulary_size: int = 100, image_vocabulary_size: int = 64,
        hidden_dim: int = 32
    ) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(text_vocabulary_size, hidden_dim)
        self.image_embedding = nn.Embedding(image_vocabulary_size, hidden_dim)
        self.decoder = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.image_token_head = nn.Linear(hidden_dim, image_vocabulary_size)

    def forward_with_shapes(self, text_ids: Tensor, image_token_ids: Tensor) -> dict[str, Tensor]:
        text_tokens = self.text_embedding(text_ids)  # [N, Lt] -> [N, Lt, D]
        text_context = text_tokens.mean(dim=1)  # [N, D]
        image_tokens = self.image_embedding(image_token_ids)  # [N, Li] -> [N, Li, D]
        condition = text_context[:, None].expand(-1, image_tokens.shape[1], -1)  # [N, Li, D]
        decoder_input = torch.cat((image_tokens, condition), dim=-1)  # [N, Li, 2D]
        decoder_states, _ = self.decoder(decoder_input)  # [N, Li, D]
        logits = self.image_token_head(decoder_states)  # [N, Li, image_vocab]
        return {
            "text_tokens": text_tokens,
            "text_context": text_context,
            "image_tokens": image_tokens,
            "decoder_input": decoder_input,
            "decoder_states": decoder_states,
            "image_token_logits": logits,
        }

    def forward(self, text_ids: Tensor, image_token_ids: Tensor) -> Tensor:
        return self.forward_with_shapes(text_ids, image_token_ids)["image_token_logits"]


class LatentDiffusionDenoiser(nn.Module):
    """Predict noise in a small image latent under a text condition.

    Inputs:
        noisy_latents: `[batch, latent_channels, latent_height, latent_width]`.
        text_embeddings: `[batch, text_dim]`.
        timesteps: integer diffusion steps `[batch]`.
    Output:
        predicted noise with the same shape and dtype as `noisy_latents`.

    Stable Diffusion uses a much larger UNet, CLIP text encoder, scheduler, and
    VAE. This module isolates the key contract: `epsilon_theta(z_t, text, t)`.
    """

    def __init__(self, latent_channels: int = 4, text_dim: int = 24, hidden_dim: int = 32) -> None:
        super().__init__()
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.time_embedding = nn.Embedding(1000, hidden_dim)
        self.input_conv = nn.Conv2d(latent_channels, hidden_dim, 3, padding=1)
        self.denoising_blocks = nn.Sequential(
            nn.GroupNorm(4, hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.SiLU(),
        )
        self.output_conv = nn.Conv2d(hidden_dim, latent_channels, 3, padding=1)

    def forward_with_shapes(
        self, noisy_latents: Tensor, text_embeddings: Tensor, timesteps: Tensor
    ) -> dict[str, Tensor]:
        text_condition = self.text_projection(text_embeddings)  # [N, text_dim] -> [N, D]
        time_condition = self.time_embedding(timesteps)  # [N] -> [N, D]
        condition = (text_condition + time_condition)[:, :, None, None]  # [N, D, 1, 1]
        latent_features = self.input_conv(noisy_latents)  # [N, C, H, W] -> [N, D, H, W]
        conditioned = latent_features + condition  # broadcast text/time over spatial axes
        denoised_features = self.denoising_blocks(conditioned)  # [N, D, H, W]
        predicted_noise = self.output_conv(denoised_features)  # [N, C, H, W]
        return {
            "noisy_latents": noisy_latents,
            "text_condition": text_condition,
            "time_condition": time_condition,
            "conditioned_latents": conditioned,
            "denoised_features": denoised_features,
            "predicted_noise": predicted_noise,
        }

    def forward(self, noisy_latents: Tensor, text_embeddings: Tensor, timesteps: Tensor) -> Tensor:
        return self.forward_with_shapes(noisy_latents, text_embeddings, timesteps)["predicted_noise"]


if __name__ == "__main__":
    torch.manual_seed(0)
    autoregressive = AutoregressiveImageTokenModel().eval()
    text_ids = torch.randint(0, 100, (2, 6))
    image_ids = torch.randint(0, 64, (2, 16))  # e.g. a 4x4 token grid flattened to length 16
    with torch.no_grad():
        ar_steps = autoregressive.forward_with_shapes(text_ids, image_ids)
    print("AUTOREGRESSIVE IMAGE TOKENS")
    for name, tensor in ar_steps.items():
        print(f"{name:20} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    diffusion = LatentDiffusionDenoiser().eval()
    latents = torch.randn(2, 4, 8, 8)
    text_embeddings = torch.randn(2, 24)
    timesteps = torch.tensor([900, 500], dtype=torch.long)
    with torch.no_grad():
        diffusion_steps = diffusion.forward_with_shapes(latents, text_embeddings, timesteps)
    print("\nLATENT DIFFUSION DENOISING")
    for name, tensor in diffusion_steps.items():
        print(f"{name:20} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
