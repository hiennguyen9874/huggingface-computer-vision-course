"""RMT concepts: Manhattan Self-Attention (MaSA) and decomposed MaSA.

MaSA adds a learned distance decay to global 2-D attention. Decomposed MaSA
mixes rows and columns separately, avoiding one full `(H*W) x (H*W)` score map.
This is a compact RMT-style classifier, not the complete paper architecture.

Run: uv run --extra cpu python codes/unit13/rmt.py
Notation: N=batch, H/W=token grid, D=width, M=heads, L=H*W.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ManhattanSelfAttention(nn.Module):
    """Global attention with distance penalty `-rate*(|dy|+|dx|)`.

    Input/output: token grid `[N,H,W,D]`. Attention weights: `[N,M,L,L]`.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.log_decay_rate = nn.Parameter(torch.full((num_heads,), -2.0))

    def forward_with_attention(self, grid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if grid.ndim != 4 or grid.shape[-1] != self.embed_dim:
            raise ValueError(f"expected [N,H,W,{self.embed_dim}], got {tuple(grid.shape)}")
        batch, height, width, _ = grid.shape
        length = height * width
        tokens = grid.reshape(batch, length, self.embed_dim)                # [N,L,D]
        qkv = self.qkv(tokens).view(batch, length, 3, self.num_heads,
                                    self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                             # each [N,M,L,Dh]
        scores = torch.einsum("nmld,nmsd->nmls", q, k) / self.head_dim ** 0.5
        rows, columns = torch.meshgrid(torch.arange(height, device=grid.device),
                                       torch.arange(width, device=grid.device),
                                       indexing="ij")
        coordinates = torch.stack((rows.flatten(), columns.flatten()), dim=-1) # [L,2]
        distance = (coordinates[:, None] - coordinates[None]).abs().sum(-1)    # [L,L]
        rates = torch.nn.functional.softplus(self.log_decay_rate)              # [M]
        spatial_bias = -rates[:, None, None] * distance[None]                   # [M,L,L]
        attention = torch.softmax(scores + spatial_bias[None], dim=-1)          # [N,M,L,L]
        heads = torch.einsum("nmls,nmsd->nmld", attention, v)                   # [N,M,L,Dh]
        mixed = heads.transpose(1, 2).reshape(batch, length, self.embed_dim)
        output = self.output(mixed).reshape(batch, height, width, self.embed_dim)
        return output, attention, distance

    def forward(self, grid: Tensor) -> Tensor:
        return self.forward_with_attention(grid)[0]


class AxisAttention(nn.Module):
    """Attention over one axis with learned 1-D distance decay.

    Input/output `[B,S,D]`; B may combine image batch and rows or columns.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        self.log_decay_rate = nn.Parameter(torch.full((num_heads,), -2.0))

    def forward(self, sequences: Tensor) -> Tensor:
        batch, length, width = sequences.shape
        qkv = self.qkv(sequences).view(batch, length, 3, self.num_heads,
                                       self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                             # [B,M,S,Dh]
        scores = torch.einsum("bmld,bmsd->bmls", q, k) / self.head_dim ** 0.5
        positions = torch.arange(length, device=sequences.device)
        distance = (positions[:, None] - positions[None]).abs()             # [S,S]
        bias = -torch.nn.functional.softplus(self.log_decay_rate)[:, None, None] * distance
        attention = torch.softmax(scores + bias[None], dim=-1)              # [B,M,S,S]
        heads = torch.einsum("bmls,bmsd->bmld", attention, v)
        return self.output(heads.transpose(1, 2).reshape(batch, length, width))


class DecomposedManhattanAttention(nn.Module):
    """Row mixing followed by column mixing on `[N,H,W,D]`.

    Largest attention tensors are `[N*H,M,W,W]` and `[N*W,M,H,H]`, rather
    than global MaSA's `[N,M,H*W,H*W]`.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.horizontal = AxisAttention(embed_dim, num_heads)
        self.vertical = AxisAttention(embed_dim, num_heads)

    def forward_with_shapes(self, grid: Tensor) -> dict[str, Tensor]:
        if grid.ndim != 4 or grid.shape[-1] != self.embed_dim:
            raise ValueError(f"expected [N,H,W,{self.embed_dim}], got {tuple(grid.shape)}")
        batch, height, width, channels = grid.shape
        rows = grid.reshape(batch * height, width, channels)                 # [N*H,W,D]
        mixed_rows = self.horizontal(rows).reshape(batch, height, width, channels)
        columns = mixed_rows.permute(0, 2, 1, 3).reshape(batch * width,
                                                         height, channels)   # [N*W,H,D]
        mixed_columns = self.vertical(columns).reshape(batch, width, height,
                                                        channels).permute(0, 2, 1, 3)
        return {"input_grid": grid, "row_sequences": rows,
                "row_mixed_grid": mixed_rows, "column_sequences": columns,
                "output_grid": mixed_columns}

    def forward(self, grid: Tensor) -> Tensor:
        return self.forward_with_shapes(grid)["output_grid"]


class TinyRMT(nn.Module):
    """RGB `[N,3,H,W]` -> patch grid -> MaSA(D) -> logits `[N,K]`."""

    def __init__(self, embed_dim: int = 48, patch_size: int = 8,
                 num_classes: int = 10, decomposed: bool = True) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.mixer = (DecomposedManhattanAttention(embed_dim) if decomposed
                      else ManhattanSelfAttention(embed_dim))
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected [N,3,H,W], got {tuple(images.shape)}")
        patch_map = self.patch_embedding(images)                             # [N,D,H/P,W/P]
        grid = patch_map.permute(0, 2, 3, 1)                                # [N,H/P,W/P,D]
        mixed = grid + self.mixer(self.norm(grid))                           # same shape
        pooled = mixed.mean(dim=(1, 2))                                     # [N,D]
        logits = self.head(pooled)                                          # [N,K]
        return {"images": images, "patch_map": patch_map, "token_grid": grid,
                "mixed_grid": mixed, "pooled": pooled, "logits": logits}

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    images = torch.randn(2, 3, 64, 64)  # 8x8 patch grid
    model = TinyRMT(decomposed=True).eval()
    with torch.no_grad():
        trace = model.forward_with_shapes(images)
        decomposition = model.mixer.forward_with_shapes(trace["token_grid"])
        global_output, global_attention, distance = ManhattanSelfAttention(48).eval().forward_with_attention(
            trace["token_grid"]
        )
    print("RMT classifier with decomposed MaSA")
    for name, value in trace.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("\nDecomposed axis tensors")
    for name, value in decomposition.items():
        print(f"{name:20} shape={tuple(value.shape)}")
    print("\nGlobal MaSA comparison")
    print("global output       shape=", tuple(global_output.shape))
    print("Manhattan distance  shape=", tuple(distance.shape))
    print("global attention    shape=", tuple(global_attention.shape))
