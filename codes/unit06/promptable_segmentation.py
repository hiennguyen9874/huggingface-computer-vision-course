"""A compact SAM-style promptable segmentation model.

Run: uv run python codes/unit06/promptable_segmentation.py
This is an architectural lesson, not Meta's pretrained SAM. It demonstrates the
three modules: image encoder, point/box prompt encoder, and mask decoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ImageEncoder(nn.Module):
    """Encode RGB `[N,3,H,W]` into dense features `[N,D,H/4,W/4]`."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1), nn.GELU(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.layers(images)


class PromptEncoder(nn.Module):
    """Encode normalized point `[x,y]` and box `[xmin,ymin,xmax,ymax]` prompts."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.point_coordinates = nn.Linear(2, hidden_dim)
        self.point_label = nn.Embedding(2, hidden_dim)  # 0=negative, 1=positive point
        self.box_coordinates = nn.Linear(4, hidden_dim)
        self.box_type = nn.Parameter(torch.randn(hidden_dim))

    def forward(
        self, points: Tensor | None, point_labels: Tensor | None, boxes: Tensor | None
    ) -> Tensor:
        """Return prompt tokens `[N,T,D]`, where `T=P` plus one optional box."""
        tokens: list[Tensor] = []
        if points is not None:
            if point_labels is None or points.shape[:2] != point_labels.shape or points.shape[-1] != 2:
                raise ValueError("points [N,P,2] require matching point_labels [N,P]")
            tokens.append(self.point_coordinates(points) + self.point_label(point_labels.long()))
        if boxes is not None:
            if boxes.ndim != 2 or boxes.shape[-1] != 4:
                raise ValueError("boxes must have normalized shape [N,4]")
            tokens.append((self.box_coordinates(boxes) + self.box_type).unsqueeze(1))
        if not tokens:
            raise ValueError("provide at least one point or box prompt")
        return torch.cat(tokens, dim=1)


class MaskDecoder(nn.Module):
    """Cross-attend mask queries to image features and emit `M` mask candidates."""

    def __init__(self, hidden_dim: int, num_masks: int) -> None:
        super().__init__()
        self.mask_tokens = nn.Embedding(num_masks, hidden_dim)             # [M,D]
        layer = nn.TransformerDecoderLayer(
            hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=0.0, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.feature_projection = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)

    def forward(self, image_features: Tensor, prompt_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, channels, height, width = image_features.shape
        image_tokens = image_features.flatten(2).transpose(1, 2)           # [N,h*w,D]
        # Every candidate sees the same prompt summary but has a different learned mask token.
        prompt_summary = prompt_tokens.mean(dim=1, keepdim=True)           # [N,1,D]
        mask_queries = self.mask_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        mask_queries = mask_queries + prompt_summary                       # [N,M,D]
        decoded_queries = self.decoder(mask_queries, image_tokens)         # [N,M,D]
        dense_features = self.feature_projection(image_features)           # [N,D,h,w]
        low_resolution_masks = torch.einsum("nmd,ndhw->nmhw", decoded_queries, dense_features)
        quality_scores = self.quality_head(decoded_queries).squeeze(-1).sigmoid() # [N,M]
        return image_tokens, low_resolution_masks, quality_scores


class PromptableSegmenter(nn.Module):
    """Map image + spatial prompt to candidate mask logits and quality scores."""

    def __init__(self, hidden_dim: int = 64, num_masks: int = 3) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(hidden_dim)
        self.prompt_encoder = PromptEncoder(hidden_dim)
        self.mask_decoder = MaskDecoder(hidden_dim, num_masks)

    def forward_with_shapes(
        self, images: Tensor, points: Tensor | None = None,
        point_labels: Tensor | None = None, boxes: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images [N,3,H,W], got {tuple(images.shape)}")
        image_features = self.image_encoder(images)                        # [N,D,H/4,W/4]
        prompt_tokens = self.prompt_encoder(points, point_labels, boxes)   # [N,T,D]
        image_tokens, low_res_masks, quality = self.mask_decoder(image_features, prompt_tokens)
        mask_logits = F.interpolate(low_res_masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
        masks = mask_logits > 0                                            # [N,M,H,W], bool
        return {
            "images": images, "image_features": image_features,
            "image_tokens": image_tokens, "prompt_tokens": prompt_tokens,
            "low_resolution_mask_logits": low_res_masks,
            "mask_logits": mask_logits, "masks": masks, "quality_scores": quality,
        }

    def forward(
        self, images: Tensor, points: Tensor | None = None,
        point_labels: Tensor | None = None, boxes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        steps = self.forward_with_shapes(images, points, point_labels, boxes)
        return steps["mask_logits"], steps["quality_scores"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = PromptableSegmenter(hidden_dim=64, num_masks=3).eval()
    images = torch.rand(2, 3, 64, 80)
    points = torch.tensor([[[0.25, 0.40], [0.80, 0.70]], [[0.50, 0.50], [0.10, 0.10]]])
    point_labels = torch.tensor([[1, 0], [1, 0]])
    boxes = torch.tensor([[0.10, 0.20, 0.60, 0.80], [0.30, 0.30, 0.90, 0.90]])
    with torch.no_grad():
        steps = model.forward_with_shapes(images, points, point_labels, boxes)
    for name, value in steps.items():
        print(f"{name:28} shape={tuple(value.shape)}, dtype={value.dtype}")
    best_candidate = steps["quality_scores"].argmax(dim=1)                 # [N]
    print(f"best mask candidate per image: {best_candidate.tolist()}")
