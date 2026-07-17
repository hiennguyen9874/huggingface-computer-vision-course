"""OWL-ViT-style open-vocabulary detection from patch and text embeddings.

Run with:
    uv run python codes/unit04/owl_vit.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EducationalOWLVit(nn.Module):
    """Score every image patch against free-form text-query embeddings.

    Inputs:
        images: float RGB tensor `[batch, 3, 32, 32]`.
        query_embeddings: float text features `[batch, queries, text_dim]`.
    Outputs:
        query logits `[batch, patches, queries]` and normalized boxes
        `[batch, patches, 4]` in `(center_x, center_y, width, height)` format.

    A real OWL-ViT obtains query embeddings from its text Transformer. Accepting
    them explicitly keeps this file focused on detection fine-tuning mechanics.
    """

    def __init__(self, text_dim: int = 24, hidden_dim: int = 32) -> None:
        super().__init__()
        self.patch_encoder = nn.Conv2d(3, hidden_dim, kernel_size=8, stride=8)
        self.object_projection = nn.Linear(hidden_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4)
        )
        self.objectness_head = nn.Linear(hidden_dim, 1)

    def forward_with_shapes(self, images: Tensor, query_embeddings: Tensor) -> dict[str, Tensor]:
        patch_grid = self.patch_encoder(images)  # [N, 3, 32, 32] -> [N, D, 4, 4]
        patch_tokens = patch_grid.flatten(2).transpose(1, 2)  # [N, 16, D]
        objects = F.normalize(self.object_projection(patch_tokens), dim=-1)  # [N, 16, D]
        queries = F.normalize(self.text_projection(query_embeddings), dim=-1)  # [N, Q, D]
        query_logits = objects @ queries.transpose(1, 2)  # [N, 16, Q]
        objectness = self.objectness_head(patch_tokens)  # [N, 16, 1]
        boxes = self.box_head(patch_tokens).sigmoid()  # normalized cx, cy, w, h [N, 16, 4]
        return {
            "images": images,
            "query_embeddings": query_embeddings,
            "patch_grid": patch_grid,
            "patch_tokens": patch_tokens,
            "object_embeddings": objects,
            "projected_queries": queries,
            "query_logits": query_logits,
            "objectness_logits": objectness,
            "normalized_boxes": boxes,
        }

    def forward(self, images: Tensor, query_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        steps = self.forward_with_shapes(images, query_embeddings)
        return steps["query_logits"], steps["normalized_boxes"]


def cxcywh_to_xyxy(boxes: Tensor, image_sizes: Tensor) -> Tensor:
    """Scale normalized boxes to pixel `(x_min, y_min, x_max, y_max)` boxes.

    Inputs:
        boxes: `[batch, patches, 4]` normalized center-format boxes.
        image_sizes: `[batch, 2]` containing `(height, width)`.
    Output: pixel boxes `[batch, patches, 4]`.
    """
    center_x, center_y, width, height = boxes.unbind(dim=-1)
    xyxy = torch.stack(
        (center_x - width / 2, center_y - height / 2,
         center_x + width / 2, center_y + height / 2), dim=-1
    ).clamp(0, 1)
    scale = torch.stack(
        (image_sizes[:, 1], image_sizes[:, 0], image_sizes[:, 1], image_sizes[:, 0]), dim=-1
    )  # [N, 4] ordered width, height, width, height
    return xyxy * scale[:, None, :]  # [N, patches, 4]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = EducationalOWLVit().eval()
    images = torch.randn(2, 3, 32, 32)
    queries = torch.randn(2, 3, 24)  # three open-vocabulary text queries per image
    with torch.no_grad():
        steps = model.forward_with_shapes(images, queries)
        pixel_boxes = cxcywh_to_xyxy(
            steps["normalized_boxes"], torch.tensor([[32, 32], [32, 32]])
        )
    for name, tensor in steps.items():
        print(f"{name:20} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"pixel_boxes          shape={tuple(pixel_boxes.shape)}, dtype={pixel_boxes.dtype}")
    best_patch = steps["query_logits"][0, :, 0].argmax()
    print("best box for image 0/query 0:", pixel_boxes[0, best_patch].round().tolist())
