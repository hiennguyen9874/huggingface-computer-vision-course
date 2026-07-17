"""Vanilla NeRF: camera rays, positional encoding, MLP, and volume rendering.

The demo renders a tiny ray batch with an untrained field. It is intentionally
a transparent implementation of the core differentiable pipeline rather than
a complete scene-training system.

Run: uv run python codes/unit08/nerf.py
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def positional_encoding(x: Tensor, frequencies: int) -> Tensor:
    """Map float `[...,D]` to `[...,D*(1+2*frequencies)]` Fourier features."""
    scales = 2.0 ** torch.arange(frequencies, dtype=x.dtype, device=x.device)
    angles = x.unsqueeze(-1) * scales * math.pi
    return torch.cat((x, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)


def camera_rays(height: int, width: int, intrinsics: Tensor,
                camera_to_world: Tensor) -> tuple[Tensor, Tensor]:
    """Create +Z-forward rays: origins/directions each `[H*W,3]`.

    `intrinsics` is `[3,3]`; `camera_to_world` is `[4,4]`. Pixel centers are
    back-projected with `K^-1`, rotated into world coordinates, and normalized.
    """
    if intrinsics.shape != (3, 3) or camera_to_world.shape != (4, 4):
        raise ValueError("expected intrinsics [3,3] and camera_to_world [4,4]")
    rows, columns = torch.meshgrid(torch.arange(height, dtype=intrinsics.dtype,
                                                device=intrinsics.device) + 0.5,
                                   torch.arange(width, dtype=intrinsics.dtype,
                                                device=intrinsics.device) + 0.5,
                                   indexing="ij")
    pixels = torch.stack((columns, rows, torch.ones_like(columns)), dim=-1).reshape(-1, 3)
    camera_directions = (torch.linalg.inv(intrinsics) @ pixels.T).T
    world_directions = (camera_to_world[:3, :3] @ camera_directions.T).T
    directions = F.normalize(world_directions, dim=-1)
    origins = camera_to_world[:3, 3].expand_as(directions)
    return origins, directions


def sample_rays(origins: Tensor, directions: Tensor, near: float, far: float,
                samples: int) -> tuple[Tensor, Tensor]:
    """Uniformly sample rays `[R,3]` into positions `[R,S,3]` and depths `[R,S]`."""
    if near <= 0 or far <= near or samples < 2:
        raise ValueError("require 0 < near < far and at least two samples")
    depths = torch.linspace(near, far, samples, dtype=origins.dtype,
                            device=origins.device).expand(origins.shape[0], samples)
    points = origins[:, None, :] + depths[..., None] * directions[:, None, :]
    return points, depths


class TinyNeRF(nn.Module):
    """Continuous field `(XYZ, direction) -> (RGB, density)` for `[...,3]` inputs."""

    def __init__(self, hidden_dim: int = 64, xyz_frequencies: int = 6,
                 direction_frequencies: int = 3) -> None:
        super().__init__()
        self.xyz_frequencies = xyz_frequencies
        self.direction_frequencies = direction_frequencies
        xyz_dim = 3 * (1 + 2 * xyz_frequencies)
        direction_dim = 3 * (1 + 2 * direction_frequencies)
        self.geometry = nn.Sequential(nn.Linear(xyz_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.density_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Sequential(nn.Linear(hidden_dim + direction_dim, hidden_dim // 2),
                                        nn.ReLU(), nn.Linear(hidden_dim // 2, 3))

    def forward(self, points: Tensor, directions: Tensor) -> tuple[Tensor, Tensor]:
        """Return bounded RGB `[...,3]` and nonnegative density `[...,1]`."""
        if points.shape != directions.shape or points.shape[-1] != 3:
            raise ValueError("points and directions must have matching shape [...,3]")
        xyz_encoded = positional_encoding(points, self.xyz_frequencies)
        direction_encoded = positional_encoding(F.normalize(directions, dim=-1),
                                                  self.direction_frequencies)
        geometry = self.geometry(xyz_encoded)
        density = F.softplus(self.density_head(geometry))
        rgb = torch.sigmoid(self.color_head(torch.cat((geometry, direction_encoded), dim=-1)))
        return rgb, density


def volume_render(rgb: Tensor, density: Tensor, depths: Tensor,
                  white_background: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """Composite samples into `(ray_rgb [R,3], depth [R], weights [R,S])`.

    Inputs are RGB `[R,S,3]`, density `[R,S,1]`, and increasing depths `[R,S]`.
    Alpha is absorption in each interval; transmittance is light surviving all
    prior intervals. Every operation is differentiable for NeRF training.
    """
    if rgb.shape[:2] != depths.shape or density.shape != (*depths.shape, 1):
        raise ValueError("expected RGB [R,S,3], density [R,S,1], depths [R,S]")
    intervals = depths[:, 1:] - depths[:, :-1]
    intervals = torch.cat((intervals, torch.full_like(intervals[:, :1], 1e3)), dim=1)
    alpha = 1.0 - torch.exp(-density[..., 0] * intervals)
    transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:, :1]),
                                            1.0 - alpha + 1e-10), dim=1), dim=1)[:, :-1]
    weights = transmittance * alpha
    rendered_rgb = (weights[..., None] * rgb).sum(dim=1)
    accumulated = weights.sum(dim=1)
    if white_background:
        rendered_rgb = rendered_rgb + (1.0 - accumulated[..., None])
    rendered_depth = (weights * depths).sum(dim=1) / accumulated.clamp_min(1e-8)
    return rendered_rgb, rendered_depth, weights


if __name__ == "__main__":
    torch.manual_seed(0)
    height, width, samples = 4, 6, 8
    intrinsics = torch.tensor([[8.0, 0.0, width / 2], [0.0, 8.0, height / 2],
                               [0.0, 0.0, 1.0]])
    origins, directions = camera_rays(height, width, intrinsics, torch.eye(4))
    points, depths = sample_rays(origins, directions, near=1.0, far=4.0, samples=samples)
    sample_directions = directions[:, None, :].expand_as(points)
    model = TinyNeRF().eval()
    with torch.no_grad():
        sample_rgb, density = model(points, sample_directions)
        rendered_rgb, rendered_depth, weights = volume_render(sample_rgb, density, depths)
    image = rendered_rgb.reshape(height, width, 3)
    depth_map = rendered_depth.reshape(height, width)
    trace = {"ray_origins": origins, "ray_directions": directions,
             "sample_depths": depths, "sample_points": points,
             "sample_rgb": sample_rgb, "sample_density": density,
             "render_weights": weights, "rendered_rays": rendered_rgb,
             "rgb_image": image, "depth_map": depth_map}
    for name, value in trace.items():
        print(f"{name:16} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("reconstruction loss example:", F.mse_loss(image, torch.rand_like(image)).item())
