"""Swin Transformer: window attention, shifted windows, and patch merging.

Run: uv run python codes/unit03/swin_transformer.py
Uses `[N, H, W, C]` inside Swin because window partitioning is clearest there.
"""

from __future__ import annotations
import torch
from torch import Tensor, nn


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """`[N,H,W,C] -> [N*num_windows, window_size**2,C]` without mixing windows."""
    n, h, w, c = x.shape
    if h % window_size or w % window_size:
        raise ValueError("H and W must be divisible by window_size")
    return (x.view(n, h // window_size, window_size, w // window_size, window_size, c)
             .permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size**2, c))


def window_reverse(windows: Tensor, window_size: int, h: int, w: int) -> Tensor:
    """Inverse partition: `[N*nW,M*M,C] -> [N,H,W,C]`."""
    windows_per_image = (h // window_size) * (w // window_size)
    n, c = windows.shape[0] // windows_per_image, windows.shape[-1]
    return (windows.view(n, h // window_size, w // window_size, window_size, window_size, c)
            .permute(0, 1, 3, 2, 4, 5).reshape(n, h, w, c))


class SwinBlock(nn.Module):
    """Window MSA block. shift=M/2 lets adjacent layers connect old windows.

    Input/output: `[N,H,W,C]`. This compact lesson uses cyclic shift without
    the production attention mask; it demonstrates data flow, not border masking.
    """
    def __init__(self, dim: int, window_size: int = 4, shift: int = 0) -> None:
        super().__init__()
        self.window_size, self.shift = window_size, shift
        self.norm1, self.attention = nn.LayerNorm(dim), nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        normalized = self.norm1(x)
        if self.shift:
            normalized = torch.roll(normalized, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(normalized, self.window_size)
        attended, _ = self.attention(windows, windows, windows, need_weights=False)
        merged = window_reverse(attended, self.window_size, x.shape[1], x.shape[2])
        if self.shift:
            merged = torch.roll(merged, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + merged
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """Merge each 2x2 neighborhood: `[N,H,W,C] -> [N,H/2,W/2,2C]`."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm, self.reduction = nn.LayerNorm(4 * dim), nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[1] % 2 or x.shape[2] % 2:
            raise ValueError("PatchMerging requires even H and W")
        four_neighbors = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                                    x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-1)
        return self.reduction(self.norm(four_neighbors))


class TinySwin(nn.Module):
    """Two-stage hierarchical classifier. Input `[N,3,32,32]`, logits `[N,K]`."""
    def __init__(self, num_classes: int = 10, dim: int = 32) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, 4, 4)
        self.regular = SwinBlock(dim, 4, 0)
        self.shifted = SwinBlock(dim, 4, 2)
        self.merge = PatchMerging(dim)
        self.stage2 = SwinBlock(2 * dim, 2, 0)
        self.norm, self.head = nn.LayerNorm(2 * dim), nn.Linear(2 * dim, num_classes)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        patches = self.patch_embed(images).permute(0, 2, 3, 1)  # [N,8,8,C]
        regular = self.regular(patches)                           # W-MSA
        shifted = self.shifted(regular)                           # SW-MSA
        merged = self.merge(shifted)                              # [N,4,4,2C]
        stage2 = self.stage2(merged)
        pooled = self.norm(stage2).mean((1, 2))                   # [N,2C]
        logits = self.head(pooled)
        return {"images": images, "patch_grid": patches, "window_attention": regular,
                "shifted_window_attention": shifted, "patch_merged": merged,
                "stage2": stage2, "pooled": pooled, "logits": logits}

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    with torch.no_grad(): trace = TinySwin().eval().forward_with_shapes(torch.randn(2, 3, 32, 32))
    for name, tensor in trace.items(): print(f"{name:28} {tuple(tensor.shape)}")
