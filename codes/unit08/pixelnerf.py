"""Small PixelNeRF-style image-conditioned radiance field.

An image encoder produces a feature map. Each 3D query is projected into the
source image, bilinearly samples its local feature, and combines that feature
with encoded XYZ and view direction to predict RGB+density.

Run: uv run python codes/unit08/pixelnerf.py
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def positional_encoding(x: Tensor, frequencies: int) -> Tensor:
    """Encode float `[...,D]` as `[...,D*(1+2*frequencies)]`."""
    scales = 2.0 ** torch.arange(frequencies, device=x.device, dtype=x.dtype)
    angles = x.unsqueeze(-1) * scales * math.pi
    return torch.cat((x, angles.sin().flatten(-2), angles.cos().flatten(-2)), dim=-1)


def project_to_grid(points_camera: Tensor, intrinsics: Tensor,
                    image_height: int, image_width: int) -> tuple[Tensor, Tensor]:
    """Project +Z-forward points `[N,Q,3]` to grid-sample coordinates `[N,Q,2]`.

    `intrinsics` is `[N,3,3]`. Grid coordinates are normalized to `[-1,1]`;
    boolean output `[N,Q]` marks points in front and inside the image.
    """
    image_h = torch.einsum("nij,nqj->nqi", intrinsics, points_camera)
    z = image_h[..., 2]
    safe_z = z.clamp_min(1e-8)
    pixels = image_h[..., :2] / safe_z.unsqueeze(-1)
    x_grid = 2.0 * pixels[..., 0] / (image_width - 1) - 1.0
    y_grid = 2.0 * pixels[..., 1] / (image_height - 1) - 1.0
    grid = torch.stack((x_grid, y_grid), dim=-1)
    valid = (z > 0) & (grid.abs() <= 1).all(dim=-1)
    return grid, valid


class ImageEncoder(nn.Module):
    """Extract source features `[N,3,H,W] -> [N,F,H/2,W/2]`."""

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, feature_channels, 3, padding=1), nn.ReLU())

    def forward(self, image: Tensor) -> Tensor:
        return self.layers(image)


class TinyPixelNeRF(nn.Module):
    """Predict RGB `[N,Q,3]` and density `[N,Q,1]` for Q image-conditioned queries."""

    def __init__(self, feature_channels: int = 24, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = ImageEncoder(feature_channels)
        xyz_dim = 3 * (1 + 2 * 4)
        direction_dim = 3 * (1 + 2 * 2)
        self.field = nn.Sequential(nn.Linear(feature_channels + xyz_dim + direction_dim, hidden_dim),
                                   nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, 4))

    def forward_with_shapes(self, image: Tensor, points_camera: Tensor,
                            directions: Tensor, intrinsics: Tensor) -> dict[str, Tensor]:
        """Trace source image `[N,3,H,W]` and query tensors `[N,Q,3]`."""
        if image.ndim != 4 or image.shape[1] != 3 or points_camera.shape != directions.shape:
            raise ValueError("expected image [N,3,H,W] and matching points/directions [N,Q,3]")
        features = self.encoder(image)
        # Intrinsics describe the original image, while grid_sample consumes the
        # half-resolution feature map; normalized coordinates remain identical.
        grid, valid = project_to_grid(points_camera, intrinsics,
                                      image.shape[-2], image.shape[-1])
        sampled = F.grid_sample(features, grid.unsqueeze(2), mode="bilinear",
                                align_corners=True).squeeze(-1).transpose(1, 2)
        xyz_encoded = positional_encoding(points_camera, 4)
        direction_encoded = positional_encoding(F.normalize(directions, dim=-1), 2)
        field_input = torch.cat((sampled, xyz_encoded, direction_encoded), dim=-1)
        raw = self.field(field_input)
        rgb = torch.sigmoid(raw[..., :3])
        density = F.softplus(raw[..., 3:]) * valid.unsqueeze(-1)
        return {"source_image": image, "image_features": features,
                "projected_grid": grid, "valid_queries": valid,
                "sampled_features": sampled, "encoded_xyz": xyz_encoded,
                "encoded_direction": direction_encoded, "field_input": field_input,
                "rgb": rgb, "density": density}

    def forward(self, image: Tensor, points_camera: Tensor, directions: Tensor,
                intrinsics: Tensor) -> tuple[Tensor, Tensor]:
        trace = self.forward_with_shapes(image, points_camera, directions, intrinsics)
        return trace["rgb"], trace["density"]


if __name__ == "__main__":
    torch.manual_seed(0)
    image = torch.randn(2, 3, 64, 64)
    points = torch.tensor([[[0.0, 0.0, 2.0], [0.5, 0.2, 2.0], [4.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    directions = F.normalize(points, dim=-1)
    intrinsics = torch.tensor([[[50.0, 0.0, 31.5], [0.0, 50.0, 31.5],
                                [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)
    model = TinyPixelNeRF().eval()
    with torch.no_grad():
        trace = model.forward_with_shapes(image, points, directions, intrinsics)
    for name, value in trace.items():
        print(f"{name:18} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("valid projected queries:", trace["valid_queries"].tolist())
