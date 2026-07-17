"""DiNAT: local neighborhood attention with dilation.

Run: uv run python codes/unit03/dinat.py
This explicit implementation favors readability over production speed.
"""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DilatedNeighborhoodAttention(nn.Module):
    """Attend to a KxK dilated neighborhood around every pixel.

    Input/output: `[N,C,H,W]`. Unfold produces neighborhoods
    `[N,heads,head_dim,K*K,H*W]`; complexity is O(H*W*K*K), not O((H*W)^2).
    Padding supplies zeros at image borders for this educational version.
    """
    def __init__(self, dim: int, heads: int = 4, kernel_size: int = 3, dilation: int = 2) -> None:
        super().__init__()
        if dim % heads: raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.kernel_size, self.dilation = kernel_size, dilation
        self.qkv, self.output = nn.Conv2d(dim, 3*dim, 1), nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape; locations = h*w; neighbors = self.kernel_size**2
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = q.reshape(n, self.heads, self.head_dim, locations)
        padding = self.dilation * (self.kernel_size // 2)
        def neighborhoods(t: Tensor) -> Tensor:
            unfolded = F.unfold(t, self.kernel_size, dilation=self.dilation, padding=padding)
            return unfolded.reshape(n, self.heads, self.head_dim, neighbors, locations)
        kn, vn = neighborhoods(k), neighborhoods(v)
        scores = (q.unsqueeze(3) * kn).sum(2) / math.sqrt(self.head_dim)  # [N,h,K*K,HW]
        weights = scores.softmax(dim=2)
        mixed = (weights.unsqueeze(2) * vn).sum(3).reshape(n, c, h, w)
        return self.output(mixed)


class DiNATBlock(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attention = DilatedNeighborhoodAttention(dim, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(nn.Conv2d(dim, 4*dim, 1), nn.GELU(), nn.Conv2d(4*dim, dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyDiNAT(nn.Module):
    """Alternate local dilation 1 and wider dilation 2. Input image -> class logits."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 32, 4, 4)  # `[N,3,32,32] -> [N,32,8,8]`
        self.local, self.dilated = DiNATBlock(32, 1), DiNATBlock(32, 2)
        self.head = nn.Linear(32, num_classes)

    def forward_with_shapes(self, x: Tensor) -> dict[str, Tensor]:
        patches = self.patch_embed(x); local = self.local(patches); dilated = self.dilated(local)
        pooled = dilated.mean((2, 3)); logits = self.head(pooled)
        return {"images": x, "patch_features": patches, "dilation_1": local,
                "dilation_2": dilated, "pooled": pooled, "logits": logits}

    def forward(self, x: Tensor) -> Tensor: return self.forward_with_shapes(x)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    with torch.no_grad(): trace = TinyDiNAT().eval().forward_with_shapes(torch.randn(2, 3, 32, 32))
    for name, tensor in trace.items(): print(f"{name:16} {tuple(tensor.shape)}")
    print("3x3 dilation=1 spans 3x3; dilation=2 samples across an effective 5x5 field.")
