"""Executable examples of point cloud, mesh, voxel, and SDF data.

No mesh library is needed: small typed containers make each representation's
shape and semantics explicit, while interpolation/SDF queries show how data is
consumed rather than merely stored.

Run: uv run python codes/unit08/representations_3d.py
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class PointCloud:
    """Unordered surface samples: points `[P,3]`, optional RGB `[P,3]`."""
    points: Tensor
    colors: Tensor | None = None


@dataclass(frozen=True)
class TriangleMesh:
    """Connected surface: vertices float `[V,3]`, faces int64 `[F,3]`."""
    vertices: Tensor
    faces: Tensor

    def face_vertices(self) -> Tensor:
        """Gather the three vertices of every face, returning `[F,3,3]`."""
        return self.vertices[self.faces]

    def face_normals(self) -> Tensor:
        """Compute unit normals `[F,3]` using the right-hand cross product."""
        triangles = self.face_vertices()
        normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0],
                                     triangles[:, 2] - triangles[:, 0])
        return torch.nn.functional.normalize(normals, dim=-1)


def trilinear_sample(volume: Tensor, points: Tensor) -> Tensor:
    """Sample scalar voxel volume `[D,H,W]` at normalized points `[P,3]`.

    Point order is `(x,y,z)`, each coordinate in `[0,1]`. Output `[P]` is the
    weighted interpolation of the eight surrounding voxel centers/corners.
    """
    if volume.ndim != 3 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("expected volume [D,H,W] and points [P,3]")
    depth, height, width = volume.shape
    scaled = points.clamp(0, 1) * points.new_tensor([width - 1, height - 1, depth - 1])
    low = scaled.floor().long()
    high = torch.minimum(low + 1, low.new_tensor([width - 1, height - 1, depth - 1]))
    fraction = scaled - low

    result = torch.zeros(points.shape[0], dtype=volume.dtype, device=volume.device)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                index = torch.where(points.new_tensor([dx, dy, dz]).bool(), high, low)
                weight_xyz = torch.where(points.new_tensor([dx, dy, dz]).bool(),
                                         fraction, 1.0 - fraction)
                result += weight_xyz.prod(dim=1) * volume[index[:, 2], index[:, 1], index[:, 0]]
    return result


def sphere_sdf(points: Tensor, radius: float = 1.0) -> Tensor:
    """Signed distance `[P]`: negative inside, zero on, positive outside."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points [P,3], got {tuple(points.shape)}")
    return torch.linalg.vector_norm(points, dim=-1) - radius


if __name__ == "__main__":
    vertices = torch.tensor([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
                             [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
    mesh = TriangleMesh(vertices, torch.tensor([[0, 1, 2], [0, 2, 3]]))
    cloud = PointCloud(vertices, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                               [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
    volume = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)
    queries = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    sdf_queries = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    trace = {"point_cloud_xyz": cloud.points, "point_cloud_rgb": cloud.colors,
             "mesh_vertices": mesh.vertices, "mesh_faces": mesh.faces,
             "face_vertices": mesh.face_vertices(), "face_normals": mesh.face_normals(),
             "voxel_grid": volume, "voxel_queries": queries,
             "interpolated_values": trilinear_sample(volume, queries),
             "sdf_queries": sdf_queries, "signed_distances": sphere_sdf(sdf_queries)}
    for name, value in trace.items():
        assert value is not None
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("SDF signs (inside/surface/outside):", trace["signed_distances"].tolist())
