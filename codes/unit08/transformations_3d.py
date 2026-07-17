"""Homogeneous 3D translation, scaling, rotation, and composition.

This lesson uses the OpenGL/column-vector convention: points are stored as
columns in a `[4, P]` tensor and transformed with `matrix @ points`.

Run: uv run python codes/unit08/transformations_3d.py
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


def translation(tx: float, ty: float, tz: float) -> Tensor:
    """Return a float32 translation matrix `[4,4]`."""
    matrix = torch.eye(4)
    matrix[:3, 3] = torch.tensor([tx, ty, tz])
    return matrix


def scaling(sx: float, sy: float, sz: float) -> Tensor:
    """Return a float32 axis-aligned scaling matrix `[4,4]`."""
    return torch.diag(torch.tensor([sx, sy, sz, 1.0]))


def rotation_x(angle_radians: float) -> Tensor:
    """Return `[4,4]` for a right-handed rotation around X."""
    c, s = math.cos(angle_radians), math.sin(angle_radians)
    return torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0],
                         [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]])


def rotation_y(angle_radians: float) -> Tensor:
    """Return `[4,4]` for a right-handed rotation around Y."""
    c, s = math.cos(angle_radians), math.sin(angle_radians)
    return torch.tensor([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0],
                         [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]])


def rotation_z(angle_radians: float) -> Tensor:
    """Return `[4,4]` for a right-handed rotation around Z."""
    c, s = math.cos(angle_radians), math.sin(angle_radians)
    return torch.tensor([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def to_homogeneous(points_xyz: Tensor) -> Tensor:
    """Convert float points `[P,3]` to column-form homogeneous points `[4,P]`.

    The final row contains ones, so translation affects points. Direction
    vectors instead use w=0 because translations must not affect directions.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"expected points [P,3], got {tuple(points_xyz.shape)}")
    ones = torch.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype,
                      device=points_xyz.device)
    return torch.cat((points_xyz, ones), dim=1).T


def transform_points(matrix: Tensor, points_h: Tensor) -> Tensor:
    """Apply `[4,4] @ [4,P]` and return Cartesian points `[P,3]`."""
    if matrix.shape != (4, 4) or points_h.ndim != 2 or points_h.shape[0] != 4:
        raise ValueError("expected matrix [4,4] and homogeneous points [4,P]")
    transformed = matrix @ points_h
    return (transformed[:3] / transformed[3:].clamp_min(1e-8)).T


if __name__ == "__main__":
    # Eight cube corners: P=8 points, each with XYZ coordinates.
    cube_xyz = torch.tensor([[x, y, z] for z in (-1.0, 1.0)
                             for y in (-1.0, 1.0) for x in (-1.0, 1.0)])
    cube_h = to_homogeneous(cube_xyz)
    translate = translation(1.0, 1.0, 0.0)
    scale = scaling(2.0, 0.5, 1.0)
    rotate = rotation_z(math.radians(20.0))

    # Rightmost matrix runs first: translate -> scale -> rotate.
    combined = rotate @ scale @ translate
    trace = {
        "cartesian_input": cube_xyz,
        "homogeneous": cube_h,
        "translation_matrix": translate,
        "scaling_matrix": scale,
        "rotation_matrix": rotate,
        "combined_matrix": combined,
        "translated_xyz": transform_points(translate, cube_h),
        "final_xyz": transform_points(combined, cube_h),
    }
    for name, value in trace.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("first corner:", cube_xyz[0].tolist(), "->", trace["final_xyz"][0].tolist())
