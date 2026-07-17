"""YOLOv1-shaped output, box decoding, IoU, and NMS.

Run with:
    uv run python codes/unit02/yolo.py
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor, nn
from torchvision.ops import nms


class Detection(TypedDict):
    boxes: Tensor  # [detections, 4], normalized xyxy coordinates
    scores: Tensor  # [detections]
    labels: Tensor  # [detections]


class YOLOv1Head(nn.Module):
    """A lightweight single-stage detector with the YOLOv1 output contract.

    Input: RGB tensor `[N, 3, H, W]` (the original paper uses 448x448).
    Output: `[N, S, S, B * 5 + C]` where each cell stores B boxes
    `(x, y, w, h, confidence)` and C class probabilities.
    """

    def __init__(self, grid_size: int = 7, boxes_per_cell: int = 2, num_classes: int = 20) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # H,W -> H/2,W/2
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> H/4,W/4
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> H/8,W/8
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # -> H/16,W/16
            nn.LeakyReLU(0.1, inplace=True),
        )
        output_channels = boxes_per_cell * 5 + num_classes
        self.spatial_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.prediction = nn.Conv2d(128, output_channels, kernel_size=1)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        features = self.backbone(images)  # [N, 3, H, W] -> [N, 128, H/16, W/16]
        grid_features = self.spatial_pool(features)  # -> [N, 128, S, S]
        channel_first = self.prediction(grid_features)  # -> [N, B*5+C, S, S]
        output = channel_first.permute(0, 2, 3, 1).contiguous()  # -> [N, S, S, B*5+C]
        return {
            "input": images,
            "backbone": features,
            "grid_features": grid_features,
            "channel_first_prediction": channel_first,
            "yolo_output": output,
        }

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["yolo_output"]


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Compute pairwise IoU for normalized or pixel-space xyxy boxes.

    Input: `boxes_a [M, 4]`, `boxes_b [N, 4]`.
    Output: pairwise IoU `[M, N]`.
    """
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0))
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0))
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(intersection.dtype).eps)


def decode_yolov1(output: Tensor, grid_size: int, boxes_per_cell: int, num_classes: int) -> tuple[Tensor, Tensor]:
    """Convert grid-relative predictions into image-relative boxes and scores.

    Input:
        output: raw tensor `[N, S, S, B*5+C]`.
    Output:
        boxes: `[N, S*S*B, 4]` normalized `x1,y1,x2,y2` boxes.
        class_scores: `[N, S*S*B, C]` confidence multiplied by class probability.
    """
    batch_size = output.shape[0]
    box_values = output[..., : boxes_per_cell * 5].reshape(
        batch_size, grid_size, grid_size, boxes_per_cell, 5
    )
    class_logits = output[..., boxes_per_cell * 5 : boxes_per_cell * 5 + num_classes]
    confidence = box_values[..., 4].sigmoid()  # [N, S, S, B]
    centers = box_values[..., 0:2].sigmoid()
    sizes = box_values[..., 2:4].sigmoid()

    cell_y, cell_x = torch.meshgrid(
        torch.arange(grid_size, device=output.device),
        torch.arange(grid_size, device=output.device),
        indexing="ij",
    )
    cell_offsets = torch.stack((cell_x, cell_y), dim=-1).to(output.dtype)
    centers = (centers + cell_offsets[None, :, :, None]) / grid_size
    half_sizes = sizes / 2
    boxes = torch.cat((centers - half_sizes, centers + half_sizes), dim=-1).clamp(0, 1)
    boxes = boxes.reshape(batch_size, grid_size * grid_size * boxes_per_cell, 4)

    probabilities = class_logits.softmax(dim=-1)  # [N, S, S, C]
    class_scores = confidence.unsqueeze(-1) * probabilities.unsqueeze(3)
    class_scores = class_scores.expand(-1, -1, -1, boxes_per_cell, -1)
    class_scores = class_scores.reshape(batch_size, grid_size * grid_size * boxes_per_cell, num_classes)
    return boxes, class_scores


def postprocess(
    boxes: Tensor, class_scores: Tensor, score_threshold: float = 0.05, iou_threshold: float = 0.5
) -> list[Detection]:
    """Filter low scores and run class-wise non-maximum suppression.

    Input: boxes `[N, K, 4]`, class_scores `[N, K, C]`.
    Output: one detection dictionary per batch item.
    """
    detections: list[Detection] = []
    for image_boxes, image_scores in zip(boxes, class_scores):
        scores, labels = image_scores.max(dim=-1)
        keep = scores >= score_threshold
        selected_boxes, selected_scores, selected_labels = image_boxes[keep], scores[keep], labels[keep]
        kept_indices: list[Tensor] = []
        for class_id in selected_labels.unique():
            class_indices = torch.where(selected_labels == class_id)[0]
            kept_indices.append(nms(selected_boxes[class_indices], selected_scores[class_indices], iou_threshold))
            kept_indices[-1] = class_indices[kept_indices[-1]]
        if kept_indices:
            keep = torch.cat(kept_indices)
            keep = keep[selected_scores[keep].argsort(descending=True)]
        else:
            keep = torch.empty(0, dtype=torch.long, device=boxes.device)
        detections.append(
            {"boxes": selected_boxes[keep], "scores": selected_scores[keep], "labels": selected_labels[keep]}
        )
    return detections


def yolov1_loss(
    prediction: Tensor,
    target: Tensor,
    grid_size: int,
    boxes_per_cell: int,
    num_classes: int,
    lambda_coord: float = 5.0,
    lambda_no_object: float = 0.5,
) -> Tensor:
    """Educational weighted squared-error loss matching the chapter's terms.

    The target uses the same `[N,S,S,B*5+C]` layout. A positive confidence in a
    target box marks it responsible for an object. Production YOLO losses add
    more careful assignment and parameterization; this function exposes the
    localization, objectness, and class-loss weighting described in Unit 2.
    """
    del grid_size, num_classes  # dimensions are already represented by tensor shapes
    box_end = boxes_per_cell * 5
    predicted = prediction[..., :box_end].reshape(
        prediction.shape[0], prediction.shape[1], prediction.shape[2], boxes_per_cell, 5
    )
    expected = target[..., :box_end].reshape(
        target.shape[0], target.shape[1], target.shape[2], boxes_per_cell, 5
    )
    responsible = expected[..., 4] > 0
    localization = (predicted[..., :2] - expected[..., :2]).square()
    localization = localization + (
        predicted[..., 2:4].clamp(min=0).sqrt() - expected[..., 2:4].clamp(min=0).sqrt()
    ).square()
    localization_loss = (localization * responsible.unsqueeze(-1)).sum()
    confidence_error = (predicted[..., 4] - expected[..., 4]).square()
    confidence_weight = torch.where(responsible, 1.0, lambda_no_object)
    confidence_loss = (confidence_error * confidence_weight).sum()
    class_start = boxes_per_cell * 5
    object_cells = responsible.any(dim=-1).unsqueeze(-1)
    class_loss = (
        (prediction[..., class_start:] - target[..., class_start:]).square() * object_cells
    ).sum()
    return lambda_coord * localization_loss + confidence_loss + class_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    model = YOLOv1Head().eval()
    images = torch.randn(1, 3, 448, 448)
    with torch.no_grad():
        tensors = model.forward_with_shapes(images)
    for name, tensor in tensors.items():
        print(f"{name:28} {tuple(tensor.shape)}")

    raw_output = tensors["yolo_output"]
    boxes, class_scores = decode_yolov1(raw_output, grid_size=7, boxes_per_cell=2, num_classes=20)
    print(f"decoded boxes               {tuple(boxes.shape)}")
    print(f"class-specific scores       {tuple(class_scores.shape)}")
    final = postprocess(boxes, class_scores, score_threshold=0.01)
    print(f"detections after NMS        {tuple(final[0]['boxes'].shape)}")
