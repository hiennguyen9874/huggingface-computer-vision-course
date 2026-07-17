"""MobileViT v2: CNN locality, token unfolding, and linear separable attention.

Run: uv run python codes/unit03/mobilevit_v2.py
"""

from __future__ import annotations
import torch
from torch import Tensor, nn


class SeparableSelfAttention(nn.Module):
    """MobileViTv2-style linear attention without a token-by-token score matrix.

    Input/output tokens: `[N,P,L,D]`, where P=pixels per local patch and
    L=number of patches. A scalar context score `[N,P,L,1]` weights values;
    summation over L costs O(L), unlike global self-attention's O(L^2).
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score, self.key, self.value, self.output = (
            nn.Linear(dim, 1), nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = self.score(x).softmax(dim=2)       # normalize across L patches
        context = (weights * self.key(x)).sum(dim=2, keepdim=True)  # [N,P,1,D]
        return self.output(torch.relu(self.value(x)) * context)      # broadcast over L


class MobileViTV2Block(nn.Module):
    """Preserve feature-map shape while combining local and global features.

    Input/output: `[N,C,H,W]`; H/W must be divisible by patch_size.
    Unfolded tokens are `[N,P=patch_size^2,L=(H/patch)*(W/patch),D]`.
    """
    def __init__(self, channels: int = 32, dim: int = 48, patch_size: int = 2) -> None:
        super().__init__(); self.patch_size = patch_size
        self.local = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                   nn.Conv2d(channels, dim, 1), nn.GELU())
        self.norm1, self.attention = nn.LayerNorm(dim), SeparableSelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
        self.project = nn.Conv2d(dim, channels, 1)

    def unfold(self, x: Tensor) -> Tensor:
        n, d, h, w = x.shape; p = self.patch_size
        if h % p or w % p: raise ValueError("H and W must be divisible by patch_size")
        # [N,D,H/p,p,W/p,p] -> [N,P,L,D]
        return x.view(n, d, h//p, p, w//p, p).permute(0, 3, 5, 2, 4, 1).reshape(n, p*p, -1, d)

    def fold(self, tokens: Tensor, h: int, w: int) -> Tensor:
        n, _, _, d = tokens.shape; p = self.patch_size
        return tokens.view(n, p, p, h//p, w//p, d).permute(0, 5, 3, 1, 4, 2).reshape(n, d, h, w)

    def forward_with_shapes(self, x: Tensor) -> dict[str, Tensor]:
        local = self.local(x); tokens = self.unfold(local)
        attended = tokens + self.attention(self.norm1(tokens))
        transformed = attended + self.mlp(self.norm2(attended))
        folded = self.fold(transformed, x.shape[2], x.shape[3])
        output = x + self.project(folded)
        return {"input": x, "local_features": local, "unfolded_tokens": tokens,
                "global_tokens": transformed, "folded_features": folded, "output": output}

    def forward(self, x: Tensor) -> Tensor: return self.forward_with_shapes(x)["output"]


class TinyMobileViTV2(nn.Module):
    """Input `[N,3,32,32]`; output logits `[N,num_classes]`."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU())
        self.block, self.head = MobileViTV2Block(), nn.Linear(32, num_classes)

    def forward(self, x: Tensor) -> Tensor: return self.head(self.block(self.stem(x)).mean((2, 3)))


if __name__ == "__main__":
    torch.manual_seed(0); model = TinyMobileViTV2().eval(); images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        stem = model.stem(images); trace = model.block.forward_with_shapes(stem); logits = model.head(trace["output"].mean((2,3)))
    print(f"images           {tuple(images.shape)}")
    print(f"stem             {tuple(stem.shape)}")
    for name, tensor in trace.items(): print(f"{name:16} {tuple(tensor.shape)}")
    print(f"logits           {tuple(logits.shape)}")
