"""R(2+1)D: factor each 3D convolution into spatial then temporal parts.

Run: uv run python codes/unit07/r2plus1d.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class R2Plus1DBlock(nn.Module):
    """`[N,Cin,T,H,W] -> [N,Cout,T,H/2,W/2]` with extra nonlinearity.

    `(1,3,3)` learns per-frame spatial patterns; `(3,1,1)` then mixes time.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        middle = max(in_channels, out_channels // 2)
        self.spatial = nn.Sequential(nn.Conv3d(in_channels, middle, (1,3,3), (1,2,2), (0,1,1)),
                                     nn.BatchNorm3d(middle), nn.ReLU())
        self.temporal = nn.Sequential(nn.Conv3d(middle, out_channels, (3,1,1), padding=(1,0,0)),
                                      nn.BatchNorm3d(out_channels), nn.ReLU())

    def forward_with_shapes(self, x: Tensor) -> dict[str, Tensor]:
        spatial = self.spatial(x); temporal = self.temporal(spatial)
        return {"input": x, "spatial_features": spatial, "temporal_features": temporal}

    def forward(self, x: Tensor) -> Tensor: return self.temporal(self.spatial(x))


class TinyR2Plus1D(nn.Module):
    """Classify `[N,3,T,H,W]` clips into `[N,K]` logits."""
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__(); self.block1 = R2Plus1DBlock(3, 16); self.block2 = R2Plus1DBlock(16, 32)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.block2(self.block1(x)).mean((2,3,4)))


if __name__ == "__main__":
    torch.manual_seed(0); x = torch.randn(2,3,8,32,32); model = TinyR2Plus1D().eval()
    with torch.no_grad():
        first = model.block1.forward_with_shapes(x); second = model.block2(first["temporal_features"])
        pooled = second.mean((2,3,4)); logits = model.classifier(pooled)
    for name,value in first.items(): print(f"block1 {name:17} shape={tuple(value.shape)}")
    print(f"block2 output             shape={tuple(second.shape)}")
    print(f"global pool               shape={tuple(pooled.shape)}")
    print(f"logits                    shape={tuple(logits.shape)}")
