"""Metrics for binary and multiclass image segmentation.

Run: uv run python codes/unit06/segmentation_metrics.py
Binary masks have shape `[N,H,W]`; multiclass label maps contain class IDs and
have shape `[N,H,W]`. Metrics return scalar float tensors.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _check_equal_masks(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            f"expected equal [N, H, W] masks, got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )


def binary_iou(prediction: Tensor, target: Tensor) -> Tensor:
    """Intersection/union over boolean `[N,H,W]` masks; empty/empty scores 1."""
    _check_equal_masks(prediction, target)
    predicted, expected = prediction.bool(), target.bool()
    intersection = (predicted & expected).sum(dtype=torch.float32)
    union = (predicted | expected).sum(dtype=torch.float32)
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def dice_coefficient(prediction: Tensor, target: Tensor) -> Tensor:
    """Twice intersection/total foreground area; empty/empty scores 1."""
    _check_equal_masks(prediction, target)
    predicted, expected = prediction.bool(), target.bool()
    intersection = (predicted & expected).sum(dtype=torch.float32)
    total = predicted.sum(dtype=torch.float32) + expected.sum(dtype=torch.float32)
    return torch.where(total > 0, 2 * intersection / total, torch.ones_like(total))


def pixel_accuracy(prediction: Tensor, target: Tensor) -> Tensor:
    """Fraction of equal class IDs in integer label maps `[N,H,W]`."""
    _check_equal_masks(prediction, target)
    return (prediction == target).float().mean()


def mean_class_iou(prediction: Tensor, target: Tensor, num_classes: int) -> Tensor:
    """Mean IoU over classes present in prediction or target.

    Unlike global pixel accuracy, each present class receives equal weight.
    """
    _check_equal_masks(prediction, target)
    class_scores: list[Tensor] = []
    for class_id in range(num_classes):
        predicted_class = prediction == class_id          # [N,H,W], bool
        target_class = target == class_id                 # [N,H,W], bool
        union = (predicted_class | target_class).sum()
        if union > 0:
            intersection = (predicted_class & target_class).sum()
            class_scores.append(intersection.float() / union)
    if not class_scores:
        raise ValueError("no class IDs in the requested range are present")
    return torch.stack(class_scores).mean()


if __name__ == "__main__":
    target = torch.tensor([[[0, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]]], dtype=torch.long)
    prediction = torch.tensor([[[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]], dtype=torch.long)
    print(f"label maps          shape={tuple(target.shape)}, dtype={target.dtype}")
    print(f"foreground masks    shape={tuple((target == 1).shape)}, dtype=torch.bool")
    print(f"binary IoU={binary_iou(prediction == 1, target == 1):.3f}")
    print(f"Dice={dice_coefficient(prediction == 1, target == 1):.3f}")
    print(f"pixel accuracy={pixel_accuracy(prediction, target):.3f}")
    print(f"mean class IoU={mean_class_iou(prediction, target, num_classes=2):.3f}")
