"""Early, late, and hybrid fusion for image and text features.

Run with:
    uv run python codes/unit04/multimodal_fusion.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MultimodalFusion(nn.Module):
    """Compare the three fusion strategies described in Unit 4.

    Inputs:
        image_tokens: float tensor `[batch, image_length, image_dim]`.
        text_tokens: float tensor `[batch, text_length, text_dim]`.
    Outputs:
        A dictionary whose representations all have shape `[batch, hidden_dim]`.

    Early fusion mixes projected image and text tokens in one Transformer. Late
    fusion independently pools each modality and concatenates the results.
    Hybrid fusion combines the early representation with both late features.
    """

    def __init__(self, image_dim: int, text_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.joint_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.late_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hybrid_projection = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward_with_shapes(
        self, image_tokens: Tensor, text_tokens: Tensor
    ) -> dict[str, Tensor]:
        image = self.image_projection(image_tokens)  # [N, Pi, Di] -> [N, Pi, D]
        text = self.text_projection(text_tokens)  # [N, Lt, Dt] -> [N, Lt, D]

        joint_tokens = torch.cat((image, text), dim=1)  # [N, Pi + Lt, D]
        contextualized = self.joint_encoder(joint_tokens)  # [N, Pi + Lt, D]
        early = contextualized.mean(dim=1)  # [N, D]

        image_global = image.mean(dim=1)  # [N, D]
        text_global = text.mean(dim=1)  # [N, D]
        late = self.late_projection(
            torch.cat((image_global, text_global), dim=-1)
        )  # [N, 2D] -> [N, D]
        hybrid = self.hybrid_projection(
            torch.cat((early, image_global, text_global), dim=-1)
        )  # [N, 3D] -> [N, D]
        return {
            "image_projected": image,
            "text_projected": text,
            "joint_tokens": joint_tokens,
            "early_fusion": early,
            "late_fusion": late,
            "hybrid_fusion": hybrid,
        }

    def forward(self, image_tokens: Tensor, text_tokens: Tensor) -> Tensor:
        """Return the hybrid representation `[batch, hidden_dim]`."""
        return self.forward_with_shapes(image_tokens, text_tokens)["hybrid_fusion"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MultimodalFusion(image_dim=48, text_dim=24).eval()
    image_tokens = torch.randn(2, 16, 48)  # 2 images, each represented by 16 patches
    text_tokens = torch.randn(2, 6, 24)  # 2 captions, each containing 6 tokens
    with torch.no_grad():
        steps = model.forward_with_shapes(image_tokens, text_tokens)
    for name, tensor in steps.items():
        print(f"{name:18} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
