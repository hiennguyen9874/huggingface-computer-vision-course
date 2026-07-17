"""Tiny Zero123-style viewpoint-conditioned denoising module.

This demonstrates Zero123's interfaces, not Stable Diffusion quality: source
CLIP-like image embedding is concatenated with relative camera pose, while the
encoded source latent is concatenated channel-wise with a noisy target latent.

Run: uv run python codes/unit08/zero123_conditioning.py
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SourceImageEncoder(nn.Module):
    """Encode source RGB `[N,3,H,W]` into latent `[N,4,H/4,W/4]`."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.SiLU(),
                                    nn.Conv2d(16, 4, 3, stride=2, padding=1))

    def forward(self, image: Tensor) -> Tensor:
        return self.layers(image)


class TinyZero123Denoiser(nn.Module):
    """Predict target noise `[N,4,h,w]` from image, pose, timestep, and noisy latent.

    Relative pose `[N,4]` stores `(elevation, azimuth, radius, roll)` in a chosen
    consistent unit. Timesteps `[N]` are normalized diffusion-step scalars.
    """

    def __init__(self, condition_dim: int = 32) -> None:
        super().__init__()
        self.image_encoder = SourceImageEncoder()
        self.image_embedding = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                             nn.Linear(4, 16), nn.SiLU())
        self.condition = nn.Sequential(nn.Linear(16 + 4 + 1, condition_dim), nn.SiLU(),
                                       nn.Linear(condition_dim, condition_dim))
        self.input_conv = nn.Conv2d(8, condition_dim, 3, padding=1)
        self.denoiser = nn.Sequential(nn.GroupNorm(4, condition_dim), nn.SiLU(),
                                      nn.Conv2d(condition_dim, condition_dim, 3, padding=1),
                                      nn.SiLU(), nn.Conv2d(condition_dim, 4, 3, padding=1))

    def forward_with_shapes(self, source_image: Tensor, noisy_target: Tensor,
                            relative_pose: Tensor, timestep: Tensor) -> dict[str, Tensor]:
        """Return a named trace for all conditioning and denoising steps."""
        if source_image.ndim != 4 or source_image.shape[1] != 3:
            raise ValueError("expected source image [N,3,H,W]")
        batch = source_image.shape[0]
        if noisy_target.shape[0] != batch or noisy_target.shape[1] != 4:
            raise ValueError("expected noisy target [N,4,H/4,W/4]")
        if relative_pose.shape != (batch, 4) or timestep.shape != (batch,):
            raise ValueError("expected relative pose [N,4] and timestep [N]")
        source_latent = self.image_encoder(source_image)
        if source_latent.shape != noisy_target.shape:
            raise ValueError(f"source latent {tuple(source_latent.shape)} must match noisy target "
                             f"{tuple(noisy_target.shape)}")
        image_embedding = self.image_embedding(source_latent)
        condition_input = torch.cat((image_embedding, relative_pose,
                                     timestep[:, None]), dim=1)
        condition = self.condition(condition_input)
        latent_pair = torch.cat((noisy_target, source_latent), dim=1)
        hidden = self.input_conv(latent_pair) + condition[:, :, None, None]
        predicted_noise = self.denoiser(hidden)
        return {"source_image": source_image, "source_latent": source_latent,
                "noisy_target": noisy_target, "image_embedding": image_embedding,
                "relative_pose": relative_pose, "condition_vector": condition,
                "concatenated_latents": latent_pair, "denoiser_hidden": hidden,
                "predicted_noise": predicted_noise}

    def forward(self, source_image: Tensor, noisy_target: Tensor,
                relative_pose: Tensor, timestep: Tensor) -> Tensor:
        return self.forward_with_shapes(source_image, noisy_target, relative_pose,
                                        timestep)["predicted_noise"]


if __name__ == "__main__":
    torch.manual_seed(0)
    source = torch.randn(2, 3, 64, 64)
    noisy_target = torch.randn(2, 4, 16, 16)
    relative_pose = torch.tensor([[0.0, 0.5, 0.0, 0.0], [0.2, -0.5, 0.1, 0.0]])
    timestep = torch.tensor([0.25, 0.75])
    model = TinyZero123Denoiser().eval()
    with torch.no_grad():
        trace = model.forward_with_shapes(source, noisy_target, relative_pose, timestep)
    for name, value in trace.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    target_noise = torch.randn_like(trace["predicted_noise"])
    print("training objective: MSE(predicted_noise, sampled_noise) =",
          F.mse_loss(trace["predicted_noise"], target_noise).item())
