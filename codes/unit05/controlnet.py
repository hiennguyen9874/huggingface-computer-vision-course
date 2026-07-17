"""A compact ControlNet-style conditional branch with zero convolutions.

The condition may represent edges, pose, depth, segmentation, or a scribble.
Run: uv run python codes/unit05/controlnet.py
"""

from __future__ import annotations

import copy
import torch
from torch import Tensor, nn


class NoiseBackbone(nn.Module):
    """Tiny frozen diffusion backbone: noisy `[N,4,H,W]` -> features/noise."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.input = nn.Conv2d(4, channels, 3, padding=1)
        self.middle = nn.Sequential(nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU())
        self.output = nn.Conv2d(channels, 4, 3, padding=1)

    def features(self, noisy_latents: Tensor) -> Tensor:
        hidden = self.input(noisy_latents)  # [N,4,H,W] -> [N,D,H,W]
        return hidden + self.middle(hidden)

    def forward(self, noisy_latents: Tensor) -> Tensor:
        return self.output(self.features(noisy_latents))  # [N,4,H,W]


class TinyControlNet(nn.Module):
    """Inject condition residuals while preserving the pretrained initial output.

    Inputs:
        noisy_latents: `[N,4,H,W]` diffusion state.
        condition_image: `[N,1,H,W]` structural control.
    Output:
        predicted noise `[N,4,H,W]`.
    """

    def __init__(self, pretrained_backbone: NoiseBackbone) -> None:
        super().__init__()
        self.backbone = pretrained_backbone.requires_grad_(False)
        self.control_branch = copy.deepcopy(pretrained_backbone)
        self.control_branch.output = nn.Identity()  # copied feature extractor is trainable
        self.control_branch.requires_grad_(True)
        self.condition_encoder = nn.Conv2d(1, 4, 3, padding=1)
        self.zero_convolution = nn.Conv2d(32, 32, 1)
        nn.init.zeros_(self.zero_convolution.weight)
        nn.init.zeros_(self.zero_convolution.bias)

    def forward_with_shapes(self, noisy_latents: Tensor, condition_image: Tensor) -> dict[str, Tensor]:
        base_features = self.backbone.features(noisy_latents)  # [N,32,H,W]
        encoded_condition = self.condition_encoder(condition_image)  # [N,1,H,W] -> [N,4,H,W]
        control_features = self.control_branch(encoded_condition)  # [N,32,H,W]
        control_residual = self.zero_convolution(control_features)  # starts exactly at zero
        combined_features = base_features + control_residual
        predicted_noise = self.backbone.output(combined_features)  # [N,4,H,W]
        return {"noisy_latents": noisy_latents, "condition_image": condition_image,
                "encoded_condition": encoded_condition, "base_features": base_features,
                "control_features": control_features, "control_residual": control_residual,
                "predicted_noise": predicted_noise}

    def forward(self, noisy_latents: Tensor, condition_image: Tensor) -> Tensor:
        return self.forward_with_shapes(noisy_latents, condition_image)["predicted_noise"]


if __name__ == "__main__":
    torch.manual_seed(0)
    backbone = NoiseBackbone().eval()
    model = TinyControlNet(backbone).eval()
    noisy, edges = torch.randn(2, 4, 16, 16), torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        steps = model.forward_with_shapes(noisy, edges)
        unchanged = torch.allclose(steps["predicted_noise"], backbone(noisy))
    for name, value in steps.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"zero_initialized_residual_norm={steps['control_residual'].norm():.6f}")
    print(f"initial_output_matches_frozen_backbone={unchanged}")
