"""DreamBooth's subject and prior-preservation denoising objectives.

DreamBooth fine-tunes an existing diffusion denoiser; it is a training procedure,
not a new backbone. This tiny conditioned denoiser makes that procedure runnable.
Run: uv run python codes/unit05/dreambooth.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TinyPromptDenoiser(nn.Module):
    """Predict noise `[N,4,H,W]` from noisy latents, timestep, and tokens `[N,L]`."""

    def __init__(self, vocabulary_size: int = 100, width: int = 32) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, width)
        self.time_embedding = nn.Embedding(1000, width)
        self.input = nn.Conv2d(4, width, 3, padding=1)
        self.output = nn.Sequential(nn.SiLU(), nn.Conv2d(width, 4, 3, padding=1))

    def forward_with_shapes(self, noisy_latents: Tensor, timesteps: Tensor, token_ids: Tensor) -> dict[str, Tensor]:
        prompt = self.token_embedding(token_ids).mean(1)  # [N,L,D] -> [N,D]
        time = self.time_embedding(timesteps)  # [N,D]
        image_features = self.input(noisy_latents)  # [N,4,H,W] -> [N,D,H,W]
        conditioned = image_features + (prompt + time)[:, :, None, None]
        predicted_noise = self.output(conditioned)  # [N,D,H,W] -> [N,4,H,W]
        return {"noisy_latents": noisy_latents, "prompt_embedding": prompt,
                "time_embedding": time, "conditioned_features": conditioned,
                "predicted_noise": predicted_noise}

    def forward(self, noisy_latents: Tensor, timesteps: Tensor, token_ids: Tensor) -> Tensor:
        return self.forward_with_shapes(noisy_latents, timesteps, token_ids)["predicted_noise"]


def dreambooth_loss(
    model: TinyPromptDenoiser,
    instance_batch: tuple[Tensor, Tensor, Tensor, Tensor],
    prior_batch: tuple[Tensor, Tensor, Tensor, Tensor],
    prior_weight: float = 1.0,
) -> dict[str, Tensor]:
    """Compute DreamBooth's scalar instance + class-prior denoising loss.

    Each tuple contains `(noisy_latents [N,4,H,W], timesteps [N], token_ids
    [N,L], target_noise [N,4,H,W])`. Instance prompts contain the rare subject
    identifier; class prompts omit it. The prior term reduces language drift and
    overfitting to the few subject photos.
    """
    instance_latents, instance_time, instance_tokens, instance_noise = instance_batch
    prior_latents, prior_time, prior_tokens, prior_noise = prior_batch
    instance_prediction = model(instance_latents, instance_time, instance_tokens)
    prior_prediction = model(prior_latents, prior_time, prior_tokens)
    instance_loss = F.mse_loss(instance_prediction, instance_noise)
    prior_loss = F.mse_loss(prior_prediction, prior_noise)
    return {"instance_loss": instance_loss, "prior_preservation_loss": prior_loss,
            "total_loss": instance_loss + prior_weight * prior_loss}


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyPromptDenoiser().eval()
    # Token 99 acts like the rare identifier in "a photo of sks dog".
    instance_tokens = torch.tensor([[1, 2, 99, 7], [1, 2, 99, 7]])
    prior_tokens = torch.tensor([[1, 2, 7, 0], [1, 2, 7, 0]])  # "a photo of dog"
    times = torch.tensor([100, 700])
    instance_latents, prior_latents = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    instance_noise, prior_noise = torch.randn_like(instance_latents), torch.randn_like(prior_latents)
    with torch.no_grad():
        trace = model.forward_with_shapes(instance_latents, times, instance_tokens)
        losses = dreambooth_loss(model,
            (instance_latents, times, instance_tokens, instance_noise),
            (prior_latents, times, prior_tokens, prior_noise))
    print(f"instance_token_ids   shape={tuple(instance_tokens.shape)}, rare_identifier=99")
    for name, value in trace.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print(", ".join(f"{name}={value:.4f}" for name, value in losses.items()))
