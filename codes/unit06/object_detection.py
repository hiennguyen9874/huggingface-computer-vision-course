"""A compact DETR-style object detector with explicit tensor contracts.

Run: uv run python codes/unit06/object_detection.py
This educational model uses random weights. Real `facebook/detr-resnet-50` is
larger and pretrained, but follows the same image -> query classes + boxes idea.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def center_to_corners(boxes: Tensor) -> Tensor:
    """Convert normalized `[...,cx,cy,w,h]` boxes to clamped `[...,xmin,ymin,xmax,ymax]`."""
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1).clamp(0, 1)


class TinyDETR(nn.Module):
    """Predict a fixed set of `Q` objects from RGB images `[N,3,H,W]`.

    Class output `[N,Q,K+1]` includes a final "no object" class. Box output
    `[N,Q,4]` uses normalized center format, independent of image resolution.
    """

    def __init__(self, num_classes: int, num_queries: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(), # [N,3,H,W] -> [N,32,H/2,W/2]
            nn.Conv2d(32, hidden_dim, 3, stride=2, padding=1), nn.ReLU(), # -> [N,D,H/4,W/4]
        )
        self.query_embeddings = nn.Embedding(num_queries, hidden_dim)     # [Q,D]
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=0.0, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)           # +1 means no-object
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4)
        )

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images [N, 3, H, W], got {tuple(images.shape)}")
        feature_map = self.backbone(images)                                # [N,D,h,w]
        memory = feature_map.flatten(2).transpose(1, 2)                    # [N,h*w,D]
        queries = self.query_embeddings.weight.unsqueeze(0).expand(images.shape[0], -1, -1)
        query_features = self.decoder(tgt=queries, memory=memory)          # [N,Q,D]
        class_logits = self.class_head(query_features)                     # [N,Q,K+1]
        boxes_cxcywh = self.box_head(query_features).sigmoid()             # [N,Q,4], range [0,1]
        boxes_xyxy = center_to_corners(boxes_cxcywh)                       # [N,Q,4], range [0,1]
        return {
            "images": images,
            "feature_map": feature_map,
            "memory_sequence": memory,
            "object_queries": queries,
            "query_features": query_features,
            "class_logits": class_logits,
            "boxes_cxcywh": boxes_cxcywh,
            "boxes_xyxy": boxes_xyxy,
        }

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        steps = self.forward_with_shapes(images)
        return steps["class_logits"], steps["boxes_cxcywh"]


def decode_detections(
    class_logits: Tensor, boxes_xyxy: Tensor, image_height: int, image_width: int,
    score_threshold: float = 0.25,
) -> list[dict[str, Tensor]]:
    """Remove no-object queries and scale normalized corners to image pixels."""
    probabilities = class_logits.softmax(-1)                              # [N,Q,K+1]
    scores, labels = probabilities[..., :-1].max(-1)                      # each [N,Q]
    scale = boxes_xyxy.new_tensor([image_width, image_height, image_width, image_height])
    outputs: list[dict[str, Tensor]] = []
    for batch_index in range(class_logits.shape[0]):
        keep = scores[batch_index] >= score_threshold                     # [Q], bool
        outputs.append({
            "scores": scores[batch_index, keep],                          # [detections]
            "labels": labels[batch_index, keep],                          # [detections], long
            "boxes_xyxy": boxes_xyxy[batch_index, keep] * scale,         # [detections,4], pixels
        })
    return outputs


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyDETR(num_classes=3, num_queries=6).eval()
    images = torch.rand(2, 3, 64, 80)
    with torch.no_grad():
        steps = model.forward_with_shapes(images)
        detections = decode_detections(
            steps["class_logits"], steps["boxes_xyxy"], 64, 80, score_threshold=0.0
        )
    for name, value in steps.items():
        print(f"{name:18} shape={tuple(value.shape)}, dtype={value.dtype}")
    first = detections[0]
    print(f"decoded scores     shape={tuple(first['scores'].shape)}")
    print(f"decoded labels     shape={tuple(first['labels'].shape)}")
    print(f"decoded pixel boxes shape={tuple(first['boxes_xyxy'].shape)}")
