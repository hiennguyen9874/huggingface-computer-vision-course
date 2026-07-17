"""Tubelet embedding: non-overlapping 3D video patches become tokens.

Run: uv run python codes/unit07/tubelet_embedding.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class TubeletEmbedding(nn.Module):
    """Map float video `[N,C,T,H,W]` to tokens `[N,L,D]` with Conv3d.

    A kernel `(tube,patch,patch)` is also the stride, so tubelets do not overlap.
    `L=(T/tube)*(H/patch)*(W/patch)` when dimensions divide exactly.
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 64,
                 tube_size: int = 2, patch_size: int = 8) -> None:
        super().__init__()
        self.projection = nn.Conv3d(in_channels, embed_dim,
            kernel_size=(tube_size, patch_size, patch_size),
            stride=(tube_size, patch_size, patch_size))

    def forward_with_shapes(self, video: Tensor) -> dict[str, Tensor]:
        if video.ndim != 5:
            raise ValueError(f"expected [N,C,T,H,W], got {tuple(video.shape)}")
        grid = self.projection(video)                  # [N,D,T',H',W']
        tokens = grid.flatten(2).transpose(1, 2)       # [N,L,D]
        return {"video": video, "tubelet_grid": grid, "tokens": tokens}

    def forward(self, video: Tensor) -> Tensor:
        return self.forward_with_shapes(video)["tokens"]


if __name__ == "__main__":
    torch.manual_seed(0)
    trace = TubeletEmbedding().forward_with_shapes(torch.randn(2, 3, 8, 32, 32))
    for name, value in trace.items():
        print(f"{name:14} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("L = 4 temporal * 4 height * 4 width = 64 tubelet tokens")
