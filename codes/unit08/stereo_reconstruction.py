"""Rectified stereo projection, disparity, and 3D reconstruction.

Coordinates follow the stereo chapter: +Z points forward, the right camera is
`baseline` units along +X, and matching pixels share the same vertical value.
Distance output uses the same physical unit as `baseline`.

Run: uv run python codes/unit08/stereo_reconstruction.py
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class RectifiedStereo:
    """Shared left/right calibration for a parallel, rectified camera pair."""
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float

    def project(self, points_left_camera: Tensor) -> tuple[Tensor, Tensor]:
        """Project 3D points `[P,3]` to left/right pixels, each `[P,2]`."""
        if points_left_camera.ndim != 2 or points_left_camera.shape[1] != 3:
            raise ValueError(f"expected points [P,3], got {tuple(points_left_camera.shape)}")
        x, y, z = points_left_camera.unbind(dim=1)
        if torch.any(z <= 0):
            raise ValueError("rectified stereo expects all points in front with z > 0")
        v = self.fy * y / z + self.cy
        left = torch.stack((self.fx * x / z + self.cx, v), dim=1)
        right = torch.stack((self.fx * (x - self.baseline) / z + self.cx, v), dim=1)
        return left, right

    def reconstruct(self, left_pixels: Tensor, right_pixels: Tensor) -> tuple[Tensor, Tensor]:
        """Triangulate matching pixels `[P,2]` into `(points [P,3], disparity [P])`.

        Zero/negative disparity is invalid for this camera ordering because it
        implies infinite/negative depth; mismatched vertical coordinates are
        rejected because inputs must already be rectified correspondences.
        """
        if left_pixels.shape != right_pixels.shape or left_pixels.ndim != 2 or left_pixels.shape[1] != 2:
            raise ValueError("expected matching left/right pixels with shape [P,2]")
        if not torch.allclose(left_pixels[:, 1], right_pixels[:, 1], atol=1e-4):
            raise ValueError("rectified correspondences must have equal vertical coordinates")
        disparity = left_pixels[:, 0] - right_pixels[:, 0]
        if torch.any(disparity <= 0):
            raise ValueError("disparity must be positive; check match order and calibration")
        z = self.baseline * self.fx / disparity
        x = self.baseline * (left_pixels[:, 0] - self.cx) / disparity
        y = self.baseline * self.fx * (left_pixels[:, 1] - self.cy) / (self.fy * disparity)
        return torch.stack((x, y, z), dim=1), disparity


def pairwise_distances(points: Tensor, pairs: Tensor) -> Tensor:
    """Measure Euclidean distances `[M]` for int64 index pairs `[M,2]`."""
    endpoints = points[pairs]
    return torch.linalg.vector_norm(endpoints[:, 0] - endpoints[:, 1], dim=1)


if __name__ == "__main__":
    # Baseline and XYZ are meters; focal lengths/principal point are pixels.
    rig = RectifiedStereo(fx=452.9, fy=452.9, cx=298.85, cy=245.52, baseline=0.075)
    ground_truth = torch.tensor([[-0.30, -0.05, 0.95], [-0.08, -0.07, 1.13],
                                 [0.24, -0.03, 0.67], [0.41, -0.03, 0.67]])
    left_pixels, right_pixels = rig.project(ground_truth)
    reconstructed, disparity = rig.reconstruct(left_pixels, right_pixels)
    pairs = torch.tensor([[0, 1], [2, 3]])
    trace = {"ground_truth_xyz": ground_truth, "left_pixels": left_pixels,
             "right_pixels": right_pixels, "disparity": disparity,
             "reconstructed_xyz": reconstructed,
             "pair_indices": pairs, "distances_m": pairwise_distances(reconstructed, pairs),
             "absolute_error": (reconstructed - ground_truth).abs()}
    for name, value in trace.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("disparity pixels:", disparity.tolist())
    print("maximum reconstruction error:", trace["absolute_error"].max().item())
