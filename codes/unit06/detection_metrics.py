"""Object detection metrics: box IoU, precision/recall, AP, and mAP.

Run: uv run python codes/unit06/detection_metrics.py
Boxes use float `[xmin,ymin,xmax,ymax]`. Predictions are sorted by confidence
and greedily matched to at most one ground-truth box in the same image/class.
"""

from __future__ import annotations

import torch
from torch import Tensor


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Pairwise IoU matrix `[A,B]` from valid corner boxes `[A,4]` and `[B,4]`."""
    if boxes_a.ndim != 2 or boxes_b.ndim != 2 or boxes_a.shape[1:] != (4,) or boxes_b.shape[1:] != (4,):
        raise ValueError("box_iou expects [A, 4] and [B, 4] tensors")
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])   # [A,B,2]
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)         # [A,B]
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0).prod(-1)      # [A]
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0).prod(-1)      # [B]
    union = area_a[:, None] + area_b[None, :] - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def precision_recall(
    predicted_boxes: Tensor, scores: Tensor, predicted_image_ids: Tensor,
    target_boxes: Tensor, target_image_ids: Tensor, iou_threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Return precision/recall curves `[P]` for predictions of one class."""
    if len(predicted_boxes) != len(scores) or len(scores) != len(predicted_image_ids):
        raise ValueError("each prediction needs one score and image ID")
    order = scores.argsort(descending=True)
    true_positive = torch.zeros(len(order), dtype=torch.float32)
    matched_target = torch.zeros(len(target_boxes), dtype=torch.bool)
    for rank, prediction_index in enumerate(order.tolist()):
        same_image = (target_image_ids == predicted_image_ids[prediction_index]) & ~matched_target
        candidates = same_image.nonzero().flatten()
        if len(candidates) == 0:
            continue
        overlaps = box_iou(predicted_boxes[prediction_index : prediction_index + 1], target_boxes[candidates])[0]
        best_overlap, best_local_index = overlaps.max(0)
        if best_overlap >= iou_threshold:
            target_index = candidates[best_local_index]
            matched_target[target_index] = True
            true_positive[rank] = 1.0
    cumulative_tp = true_positive.cumsum(0)
    precision = cumulative_tp / torch.arange(1, len(order) + 1, dtype=torch.float32)
    recall = cumulative_tp / max(len(target_boxes), 1)
    return precision, recall


def average_precision(precision: Tensor, recall: Tensor) -> Tensor:
    """Area under an interpolated precision-recall curve (all-points AP)."""
    if precision.numel() == 0:
        return torch.tensor(0.0)
    padded_precision = torch.cat((torch.tensor([0.0]), precision, torch.tensor([0.0])))
    padded_recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    # Precision envelope: precision at recall r is the best precision at any r' >= r.
    padded_precision = torch.flip(torch.cummax(torch.flip(padded_precision, dims=[0]), dim=0).values, dims=[0])
    changed = padded_recall[1:] != padded_recall[:-1]
    return ((padded_recall[1:] - padded_recall[:-1]) * padded_precision[1:])[changed].sum()


def mean_average_precision(
    predicted_boxes: Tensor, scores: Tensor, predicted_labels: Tensor, predicted_image_ids: Tensor,
    target_boxes: Tensor, target_labels: Tensor, target_image_ids: Tensor,
    iou_thresholds: tuple[float, ...] = (0.5,),
) -> Tensor:
    """Mean AP over classes present in ground truth and requested IoU thresholds."""
    ap_values: list[Tensor] = []
    for threshold in iou_thresholds:
        for class_id in target_labels.unique().tolist():
            prediction_mask = predicted_labels == class_id
            target_mask = target_labels == class_id
            precision, recall = precision_recall(
                predicted_boxes[prediction_mask], scores[prediction_mask], predicted_image_ids[prediction_mask],
                target_boxes[target_mask], target_image_ids[target_mask], threshold,
            )
            ap_values.append(average_precision(precision, recall))
    if not ap_values:
        raise ValueError("mAP requires at least one ground-truth object")
    return torch.stack(ap_values).mean()


if __name__ == "__main__":
    target_boxes = torch.tensor([[10, 10, 30, 30], [40, 10, 60, 30]], dtype=torch.float32)
    target_labels = torch.tensor([0, 1])
    target_image_ids = torch.tensor([0, 0])
    predicted_boxes = torch.tensor([[11, 11, 29, 29], [0, 0, 8, 8], [39, 9, 61, 31]], dtype=torch.float32)
    predicted_labels = torch.tensor([0, 0, 1])
    predicted_image_ids = torch.tensor([0, 0, 0])
    scores = torch.tensor([0.95, 0.80, 0.90])
    print(f"pairwise IoU matrix shape={tuple(box_iou(predicted_boxes, target_boxes).shape)}")
    print(box_iou(predicted_boxes, target_boxes))
    map_50 = mean_average_precision(
        predicted_boxes, scores, predicted_labels, predicted_image_ids,
        target_boxes, target_labels, target_image_ids, (0.5,),
    )
    map_coco_style = mean_average_precision(
        predicted_boxes, scores, predicted_labels, predicted_image_ids,
        target_boxes, target_labels, target_image_ids, tuple(i / 100 for i in range(50, 100, 5)),
    )
    print(f"mAP@0.50={map_50:.3f}")
    print(f"mAP@[0.50:0.95]={map_coco_style:.3f}")
