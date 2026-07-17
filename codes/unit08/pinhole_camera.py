"""Pinhole camera intrinsics and world-to-pixel projection.

The public pipeline is world `[P,3]` -> camera `[P,3]` -> homogeneous image
`[P,3]` -> pixels `[P,2]`. This example follows Blender: the camera looks
along -Z, so visible camera-space points have negative Z.

Run: uv run python codes/unit08/pinhole_camera.py
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Camera calibration in pixels; `width`/`height` define image bounds."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def matrix(self, blender_axes: bool = True) -> Tensor:
        """Return intrinsic matrix K `[3,3]` with float32 dtype.

        Negative focal entries compensate for Blender's visible negative-Z
        points, preserving the expected left/right and up/down image axes.
        """
        sign = -1.0 if blender_axes else 1.0
        return torch.tensor([[sign * self.fx, 0.0, self.cx],
                             [0.0, sign * self.fy, self.cy],
                             [0.0, 0.0, 1.0]])


def world_to_camera(points_world: Tensor, transform: Tensor) -> Tensor:
    """Transform float points `[P,3]` with column-style matrix `[4,4]`.

    `transform` is an extrinsic world-to-camera matrix, not a camera pose
    (camera-to-world matrix). Output is camera coordinates `[P,3]`.
    """
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"expected points [P,3], got {tuple(points_world.shape)}")
    if transform.shape != (4, 4):
        raise ValueError(f"expected transform [4,4], got {tuple(transform.shape)}")
    homogeneous = torch.cat((points_world, torch.ones_like(points_world[:, :1])), dim=1)
    camera_h = (transform @ homogeneous.T).T
    return camera_h[:, :3] / camera_h[:, 3:].clamp_min(1e-8)


def project_camera_points(points_camera: Tensor, intrinsics: PinholeIntrinsics,
                          blender_axes: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """Project camera points `[P,3]` to pixels `[P,2]`.

    Returns `(image_h, pixels, visible)`: homogeneous image coordinates
    `[P,3]`, pixel coordinates `[P,2]`, and boolean visibility mask `[P]`.
    Visibility checks that depth faces the camera and pixels lie in bounds.
    """
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError(f"expected camera points [P,3], got {tuple(points_camera.shape)}")
    image_h = (intrinsics.matrix(blender_axes).to(points_camera) @ points_camera.T).T
    denominator = image_h[:, 2:3]
    if torch.any(denominator.abs() < 1e-8):
        raise ValueError("cannot project a point on the camera plane (z=0)")
    pixels = image_h[:, :2] / denominator
    in_front = points_camera[:, 2] < 0 if blender_axes else points_camera[:, 2] > 0
    visible = (in_front & (pixels[:, 0] >= 0) & (pixels[:, 0] < intrinsics.width)
               & (pixels[:, 1] >= 0) & (pixels[:, 1] < intrinsics.height))
    return image_h, pixels, visible


if __name__ == "__main__":
    intrinsics = PinholeIntrinsics(400.0, 400.0, 320.0, 240.0, 640, 480)
    points_world = torch.tensor([[0.0, 0.0, -4.0], [1.0, 0.5, -4.0],
                                 [-1.0, -0.5, -2.0], [0.0, 0.0, 2.0]])
    # Camera center is at world x=1: world-to-camera subtracts 1 from x.
    world_to_cam = torch.eye(4)
    world_to_cam[0, 3] = -1.0
    points_camera = world_to_camera(points_world, world_to_cam)
    image_h, pixels, visible = project_camera_points(points_camera, intrinsics)

    trace = {"world_points": points_world, "world_to_camera": world_to_cam,
             "camera_points": points_camera, "intrinsic_K": intrinsics.matrix(),
             "homogeneous_image": image_h, "pixels": pixels, "visible": visible}
    for name, value in trace.items():
        print(f"{name:18} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("pixels:\n", pixels)
    print("visible:", visible.tolist())
