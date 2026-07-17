"""Depth normalization, scale/shift-invariant loss, and metric evaluation.

All functions accept dense float depth maps with any common leading shape and
use a boolean mask of the same shape to exclude sensor holes and invalid depth.

Run: uv run python codes/unit08/depth_objectives.py
"""
from __future__ import annotations

import torch
from torch import Tensor


def valid_depth_mask(depth: Tensor, min_depth: float = 1e-3,
                     max_depth: float = 10.0) -> Tensor:
    """Return boolean mask matching `depth`, true for finite in-range values."""
    return torch.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)


def normalize_disparity(depth: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert valid depth to disparity and normalize median/MAD per sample.

    Input is `[N,H,W]`; output has the same shape and zeros at invalid pixels.
    Median removes additive shift and mean absolute deviation removes scale.
    """
    if depth.ndim != 3 or depth.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("expected depth [N,H,W] and boolean mask of equal shape")
    normalized = torch.zeros_like(depth)
    for sample in range(depth.shape[0]):
        valid = mask[sample]
        if not valid.any():
            raise ValueError(f"sample {sample} contains no valid depth")
        disparity = depth[sample][valid].reciprocal()
        shift = disparity.median()
        scale = (disparity - shift).abs().mean().clamp_min(eps)
        normalized[sample][valid] = (disparity - shift) / scale
    return normalized


def scale_shift_invariant_l1(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Return scalar affine-invariant L1 loss between depth maps `[N,H,W]`."""
    pred_norm = normalize_disparity(prediction, mask)
    target_norm = normalize_disparity(target, mask)
    return (pred_norm[mask] - target_norm[mask]).abs().mean()


def depth_metrics(prediction: Tensor, target: Tensor, mask: Tensor,
                  eps: float = 1e-6) -> dict[str, Tensor]:
    """Compute scalar MAE, RMSE, AbsRel, and delta1 over valid pixels.

    Metric-depth inputs must use the same physical unit and positive values.
    Each returned value is a scalar float tensor suitable for logging.
    """
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("prediction, target, and mask must have equal shapes")
    pred = prediction[mask].clamp_min(eps)
    true = target[mask].clamp_min(eps)
    if pred.numel() == 0:
        raise ValueError("cannot evaluate an empty valid mask")
    error = pred - true
    ratio = torch.maximum(true / pred, pred / true)
    return {"mae": error.abs().mean(), "rmse": error.square().mean().sqrt(),
            "abs_rel": (error.abs() / true).mean(),
            "delta1": (ratio < 1.25).float().mean()}


if __name__ == "__main__":
    target = torch.tensor([[[1.0, 2.0, 0.0], [4.0, 5.0, 12.0]],
                           [[0.5, 1.0, 2.0], [3.0, float("nan"), 6.0]]])
    # First sample has scale/shift differences; second has small local errors.
    prediction = torch.tensor([[[2.0, 4.0, 1.0], [8.0, 10.0, 20.0]],
                               [[0.6, 1.1, 1.8], [3.2, 1.0, 5.8]]])
    mask = valid_depth_mask(target)
    pred_normalized = normalize_disparity(prediction, mask)
    target_normalized = normalize_disparity(target, mask)
    invariant_loss = scale_shift_invariant_l1(prediction, target, mask)
    metrics = depth_metrics(prediction, target, mask)

    trace = {"target_depth": target, "predicted_depth": prediction,
             "valid_mask": mask, "prediction_normalized": pred_normalized,
             "target_normalized": target_normalized, "invariant_l1": invariant_loss,
             **metrics}
    for name, value in trace.items():
        print(f"{name:22} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("metric values:", {name: round(value.item(), 4) for name, value in metrics.items()})
