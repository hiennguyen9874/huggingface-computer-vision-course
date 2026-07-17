"""A minimal Denoising Diffusion Probabilistic Model (DDPM).

Shows the closed-form forward process, timestep-conditioned noise prediction,
training objective, and iterative reverse sampling.
Run: uv run python codes/unit05/diffusion.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TimeConditionedDenoiser(nn.Module):
    """Predict Gaussian noise `[N,C,H,W]` from noisy image and timestep `[N]`."""

    def __init__(self, channels: int = 1, hidden_dim: int = 32, time_dim: int = 32) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.input = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.middle = nn.Sequential(nn.GroupNorm(4, hidden_dim), nn.SiLU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))
        self.output = nn.Conv2d(hidden_dim, channels, 3, padding=1)

    def timestep_embedding(self, timesteps: Tensor) -> Tensor:
        half = self.time_dim // 2
        frequencies = torch.exp(-torch.arange(half, device=timesteps.device) * (torch.log(torch.tensor(10_000.0)) / (half - 1)))
        angles = timesteps.float()[:, None] * frequencies[None, :]  # [N, time_dim/2]
        return torch.cat((angles.sin(), angles.cos()), dim=1)  # [N, time_dim]

    def forward_with_shapes(self, noisy_images: Tensor, timesteps: Tensor) -> dict[str, Tensor]:
        if timesteps.shape != (noisy_images.shape[0],):
            raise ValueError(f"expected timesteps [N], got {tuple(timesteps.shape)}")
        time_embedding = self.timestep_embedding(timesteps)
        time_features = self.time_mlp(time_embedding)[:, :, None, None]  # [N,D,1,1]
        image_features = self.input(noisy_images)  # [N,C,H,W] -> [N,D,H,W]
        conditioned = image_features + time_features  # broadcast time over H,W
        predicted_noise = self.output(self.middle(conditioned) + conditioned)
        return {"noisy_images": noisy_images, "time_embedding": time_embedding,
                "conditioned_features": conditioned, "predicted_noise": predicted_noise}

    def forward(self, noisy_images: Tensor, timesteps: Tensor) -> Tensor:
        return self.forward_with_shapes(noisy_images, timesteps)["predicted_noise"]


class DDPM:
    """Linear-noise scheduler with `T` discrete forward/reverse steps."""

    def __init__(self, steps: int = 20, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps)  # [T]
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _extract(values: Tensor, timesteps: Tensor, images: Tensor) -> Tensor:
        return values.to(images.device)[timesteps].reshape(-1, 1, 1, 1)  # [N,1,1,1]

    def add_noise(self, clean: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Sample q(x_t|x_0); inputs/outputs are `[N,C,H,W]`."""
        noise = torch.randn_like(clean) if noise is None else noise
        alpha_bar = self._extract(self.alpha_bars, timesteps, clean)
        noisy = alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise
        return noisy, noise

    def training_loss(self, model: nn.Module, clean: Tensor, timesteps: Tensor) -> Tensor:
        noisy, target_noise = self.add_noise(clean, timesteps)
        return F.mse_loss(model(noisy, timesteps), target_noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, ...], device: torch.device) -> Tensor:
        """Start at x_T Gaussian noise and repeatedly estimate x_(t-1)."""
        images = torch.randn(shape, device=device)
        for step in reversed(range(self.steps)):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            predicted_noise = model(images, timesteps)
            alpha = self.alphas[step].to(device)
            alpha_bar = self.alpha_bars[step].to(device)
            mean = (images - (1 - alpha) / (1 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
            if step > 0:
                mean = mean + self.betas[step].to(device).sqrt() * torch.randn_like(images)
            images = mean
        return images


if __name__ == "__main__":
    torch.manual_seed(0)
    model, scheduler = TimeConditionedDenoiser().eval(), DDPM(steps=8)
    clean, timesteps = torch.randn(2, 1, 16, 16), torch.tensor([1, 7])
    noisy, sampled_noise = scheduler.add_noise(clean, timesteps)
    with torch.no_grad():
        trace = model.forward_with_shapes(noisy, timesteps)
        loss = F.mse_loss(trace["predicted_noise"], sampled_noise)
        generated = scheduler.sample(model, (2, 1, 16, 16), torch.device("cpu"))
    print(f"clean_images         shape={tuple(clean.shape)}")
    print(f"sampled_noise        shape={tuple(sampled_noise.shape)}")
    for name, value in trace.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"reverse_sample       shape={tuple(generated.shape)}")
    print(f"noise_prediction_loss={loss:.4f} (untrained model: value is illustrative)")
