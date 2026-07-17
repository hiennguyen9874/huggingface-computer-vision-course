"""Tiny X3D-inspired efficient video classifier.

Run: uv run python codes/unit07/x3d.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class X3DBlock(nn.Module):
    """Bottleneck with cheap depthwise 3D convolution, `[N,C,T,H,W] -> same`."""
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__(); hidden = channels * expansion
        self.layers = nn.Sequential(
            nn.Conv3d(channels, hidden, 1), nn.ReLU(),
            nn.Conv3d(hidden, hidden, 3, padding=1, groups=hidden), nn.ReLU(),
            nn.Conv3d(hidden, channels, 1))
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor: return self.activation(x + self.layers(x))


class TinyX3D(nn.Module):
    """Classify clips `[N,3,T,H,W] -> [N,K]` using width/depth expansion."""
    def __init__(self, num_classes: int = 5, width: int = 16, depth: int = 3) -> None:
        super().__init__(); self.stem = nn.Sequential(nn.Conv3d(3,width,3,(1,2,2),1),nn.ReLU())
        self.blocks = nn.ModuleList(X3DBlock(width) for _ in range(depth)); self.head = nn.Linear(width,num_classes)

    def forward_with_shapes(self, x: Tensor) -> dict[str, Tensor]:
        trace={"input":x,"stem":self.stem(x)}; features=trace["stem"]
        for i,block in enumerate(self.blocks): features=block(features); trace[f"x3d_block_{i}"]=features
        trace["pooled"]=features.mean((2,3,4)); trace["logits"]=self.head(trace["pooled"]); return trace

    def forward(self,x:Tensor)->Tensor:return self.forward_with_shapes(x)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0); model=TinyX3D().eval()
    with torch.no_grad(): trace=model.forward_with_shapes(torch.randn(2,3,8,32,32))
    for name,value in trace.items(): print(f"{name:13} shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,} (depthwise convolution keeps this small)")
