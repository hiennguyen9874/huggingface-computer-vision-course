"""CvT: overlapping convolutional tokens and convolutional Q/K/V projections.

Run: uv run python codes/unit03/cvt.py
CvT injects CNN locality and therefore needs no explicit positional embedding.
"""

from __future__ import annotations
import math
import torch
from torch import Tensor, nn


class ConvTokenEmbedding(nn.Module):
    """Overlapping tokenization. `[N,C,H,W] -> [N,E,H',W']`."""
    def __init__(self, in_channels: int, embed_dim: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel, stride, kernel // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvolutionalAttention(nn.Module):
    """Depthwise convolutions form spatially aware Q/K/V before attention.

    Input/output feature map: `[N,C,H,W]`. Attention internally uses
    `[N,heads,H*W,head_dim]`; its score matrix is `[N,heads,H*W,H*W]`.
    """
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        if dim % heads: raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.spatial = nn.ModuleList(nn.Conv2d(dim, dim, 3, padding=1, groups=dim) for _ in range(3))
        self.channel = nn.ModuleList(nn.Conv2d(dim, dim, 1) for _ in range(3))
        self.output = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        q, k, v = [pointwise(depthwise(x)) for depthwise, pointwise in zip(self.spatial, self.channel)]
        def heads(t: Tensor) -> Tensor:
            return t.flatten(2).transpose(1, 2).reshape(n, h*w, self.heads, self.head_dim).transpose(1, 2)
        qh, kh, vh = map(heads, (q, k, v))
        weights = (qh @ kh.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(-1)
        mixed = (weights @ vh).transpose(1, 2).reshape(n, h*w, c).transpose(1, 2).reshape(n, c, h, w)
        return self.output(mixed)


class CvTStage(nn.Module):
    def __init__(self, in_channels: int, dim: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.embedding = ConvTokenEmbedding(in_channels, dim, kernel, stride)
        self.norm1, self.attention = nn.GroupNorm(1, dim), ConvolutionalAttention(dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(nn.Conv2d(dim, 4*dim, 1), nn.GELU(), nn.Conv2d(4*dim, dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyCvT(nn.Module):
    """Three-stage hierarchy. Input `[N,3,32,32]`; output logits `[N,K]`."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stage1, self.stage2, self.stage3 = CvTStage(3, 32, 7, 4), CvTStage(32, 64, 3, 2), CvTStage(64, 96, 3, 2)
        self.norm, self.head = nn.LayerNorm(96), nn.Linear(96, num_classes)

    def forward_with_shapes(self, x: Tensor) -> dict[str, Tensor]:
        s1, s2 = self.stage1(x), None
        s2 = self.stage2(s1); s3 = self.stage3(s2)
        pooled = self.norm(s3.mean((2, 3)))
        return {"images": x, "stage1": s1, "stage2": s2, "stage3": s3,
                "global_average": pooled, "logits": self.head(pooled)}

    def forward(self, x: Tensor) -> Tensor: return self.forward_with_shapes(x)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    with torch.no_grad(): trace = TinyCvT().eval().forward_with_shapes(torch.randn(2, 3, 32, 32))
    print("No position embedding: overlapping convolution supplies local spatial bias.")
    for name, tensor in trace.items(): print(f"{name:16} {tuple(tensor.shape)}")
