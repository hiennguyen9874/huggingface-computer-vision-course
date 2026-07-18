"""Hiera-style hierarchical Vision Transformer classifier.

The lesson keeps Hiera's main architectural ideas: simple absolute positions,
local mask-unit attention in early high-resolution stages, pooling between
stages, and global attention at low resolution. It intentionally omits the
paper's optimized kernels and exact production configuration.

Run: uv run --extra cpu python codes/unit13/hiera.py
Notation: N=batch, H/W=grid size, D=stage width, K=classes.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Pre-norm self-attention + MLP, `[B,L,D] -> [B,L,D]`."""

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
                                 nn.Linear(4 * embed_dim, embed_dim))

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm_1(tokens)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + attended
        return tokens + self.mlp(self.norm_2(tokens))


class MaskUnitAttention(nn.Module):
    """Apply one Transformer block independently in non-overlapping windows.

    Input/output grid: `[N,H,W,D]`; H and W must divide `window_size`.
    Each local sequence has `window_size**2` tokens.
    """

    def __init__(self, embed_dim: int, window_size: int = 2) -> None:
        super().__init__()
        self.embed_dim, self.window_size = embed_dim, window_size
        self.block = TransformerBlock(embed_dim)

    def forward_with_windows(self, grid: Tensor) -> tuple[Tensor, Tensor]:
        if grid.ndim != 4 or grid.shape[-1] != self.embed_dim:
            raise ValueError(f"expected [N,H,W,{self.embed_dim}], got {tuple(grid.shape)}")
        batch, height, width, channels = grid.shape
        window = self.window_size
        if height % window or width % window:
            raise ValueError("token-grid dimensions must be divisible by window_size")
        # [N,H,W,D] -> [N*(H/w)*(W/w),w*w,D], with no convolution.
        windows = (grid.view(batch, height // window, window, width // window,
                             window, channels)
                   .permute(0, 1, 3, 2, 4, 5)
                   .reshape(-1, window * window, channels))
        encoded_windows = self.block(windows)
        output = (encoded_windows.view(batch, height // window, width // window,
                                       window, window, channels)
                  .permute(0, 1, 3, 2, 4, 5)
                  .reshape(batch, height, width, channels))
        return output, windows

    def forward(self, grid: Tensor) -> Tensor:
        return self.forward_with_windows(grid)[0]


class PoolAndProject(nn.Module):
    """Stage transition `[N,H,W,Din] -> [N,H/2,W/2,Dout]`."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, grid: Tensor) -> Tensor:
        # Hiera removes costly vision-specific convolution here: plain max pool
        # reduces resolution, then a Linear changes channel width.
        channels_first = grid.permute(0, 3, 1, 2)                           # [N,D,H,W]
        pooled = F.max_pool2d(channels_first, kernel_size=2, stride=2)
        return self.projection(pooled.permute(0, 2, 3, 1))                  # [N,H/2,W/2,Dout]


class GlobalGridAttention(nn.Module):
    """Flatten a low-resolution grid, apply global attention, restore grid."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.block = TransformerBlock(embed_dim)

    def forward(self, grid: Tensor) -> Tensor:
        batch, height, width, channels = grid.shape
        if channels != self.embed_dim:
            raise ValueError(f"expected last dimension {self.embed_dim}, got {channels}")
        tokens = grid.reshape(batch, height * width, channels)              # [N,H*W,D]
        return self.block(tokens).reshape(batch, height, width, channels)


class TinyHiera(nn.Module):
    """Hierarchical classifier `[N,3,32,32] -> logits [N,K]`."""

    def __init__(self, image_size: int = 32, patch_size: int = 4,
                 num_classes: int = 10) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size, self.patch_size = image_size, patch_size
        grid_size = image_size // patch_size
        self.patch_embedding = nn.Conv2d(3, 32, patch_size, patch_size)
        self.absolute_position = nn.Parameter(torch.zeros(1, grid_size, grid_size, 32))
        self.stage_1 = MaskUnitAttention(32, window_size=2)
        self.transition_1 = PoolAndProject(32, 64)
        self.stage_2 = MaskUnitAttention(64, window_size=2)
        self.transition_2 = PoolAndProject(64, 128)
        self.stage_3 = GlobalGridAttention(128)
        self.norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, num_classes)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        expected = (3, self.image_size, self.image_size)
        if images.ndim != 4 or images.shape[1:] != expected:
            raise ValueError(f"expected [N,3,{self.image_size},{self.image_size}], "
                             f"got {tuple(images.shape)}")
        patch_map = self.patch_embedding(images)                             # [N,32,8,8]
        patch_grid = patch_map.permute(0, 2, 3, 1) + self.absolute_position # [N,8,8,32]
        stage_1 = self.stage_1(patch_grid)                                   # [N,8,8,32]
        pooled_1 = self.transition_1(stage_1)                               # [N,4,4,64]
        stage_2 = self.stage_2(pooled_1)                                    # [N,4,4,64]
        pooled_2 = self.transition_2(stage_2)                               # [N,2,2,128]
        stage_3 = self.stage_3(pooled_2)                                    # [N,2,2,128]
        representation = self.norm(stage_3.mean(dim=(1, 2)))                # [N,128]
        logits = self.head(representation)                                  # [N,K]
        return {"images": images, "patch_map": patch_map, "positioned_grid": patch_grid,
                "stage_1_local": stage_1, "pooled_1": pooled_1,
                "stage_2_local": stage_2, "pooled_2": pooled_2,
                "stage_3_global": stage_3, "representation": representation,
                "logits": logits}

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyHiera().eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        trace = model.forward_with_shapes(images)
        _, first_stage_windows = model.stage_1.forward_with_windows(trace["positioned_grid"])
    for name, value in trace.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("stage_1 windows      shape=", tuple(first_stage_windows.shape),
          "# [N*num_windows, tokens_per_window, D]")
