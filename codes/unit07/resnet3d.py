"""Small 3D ResNet that learns spatial and temporal filters jointly.

Run: uv run python codes/unit07/resnet3d.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class Residual3DBlock(nn.Module):
    """Residual block `[N,Cin,T,H,W] -> [N,Cout,T,H/s,W/s]`."""
    def __init__(self, in_channels: int, out_channels: int, spatial_stride: int = 1) -> None:
        super().__init__()
        stride = (1, spatial_stride, spatial_stride)  # preserve temporal resolution
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels))
        self.skip = (nn.Identity() if in_channels == out_channels and spatial_stride == 1 else
                     nn.Conv3d(in_channels, out_channels, 1, stride, bias=False))
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.main(x) + self.skip(x))


class TinyResNet3D(nn.Module):
    """Classify float clips `[N,3,T,H,W]` into logits `[N,K]`."""
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(3, 16, (3, 7, 7), (1, 2, 2), (1, 3, 3)), nn.ReLU())
        self.block1 = Residual3DBlock(16, 16)
        self.block2 = Residual3DBlock(16, 32, 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward_with_shapes(self, x: Tensor) -> dict[str, Tensor]:
        stem = self.stem(x); block1 = self.block1(stem); block2 = self.block2(block1)
        pooled = self.pool(block2).flatten(1)
        return {"input": x, "stem": stem, "residual_1": block1,
                "residual_2": block2, "pooled": pooled, "logits": self.classifier(pooled)}

    def forward(self, x: Tensor) -> Tensor: return self.forward_with_shapes(x)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0); model = TinyResNet3D().eval()
    with torch.no_grad(): trace = model.forward_with_shapes(torch.randn(2, 3, 8, 32, 32))
    for name, value in trace.items(): print(f"{name:12} shape={tuple(value.shape)}, dtype={value.dtype}")
