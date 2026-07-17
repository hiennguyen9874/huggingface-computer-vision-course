"""Tiny DPT-style encoder-decoder for monocular metric depth.

This is a runnable architecture lesson, not Depth Anything V2: small CNN stages
stand in for multi-scale ViT/DINOv2 features. The important DPT flow remains:
RGB image -> feature pyramid -> top-down fusion -> dense sigmoid depth.

Run: uv run python codes/unit08/dpt_depth.py
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvStage(nn.Module):
    """Downsample feature map `[N,Cin,H,W] -> [N,Cout,H/2,W/2]`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class FusionBlock(nn.Module):
    """Fuse same-width low/high-resolution features into `[N,C,H,W]`."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.GELU())

    def forward(self, lateral: Tensor, coarse: Tensor) -> Tensor:
        upsampled = F.interpolate(coarse, size=lateral.shape[-2:], mode="bilinear",
                                 align_corners=False)
        return self.refine(lateral + upsampled)


class TinyDPTDepth(nn.Module):
    """Map normalized RGB `[N,3,H,W]` to metric depth `[N,H,W]`.

    H and W should be divisible by 16. Output values are in `(0,max_depth)`;
    invalid sensor pixels are a target-data concern and must be masked in loss.
    """

    def __init__(self, feature_width: int = 32, max_depth: float = 10.0) -> None:
        super().__init__()
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")
        self.max_depth = max_depth
        self.stage1 = ConvStage(3, 16)       # H/2
        self.stage2 = ConvStage(16, 32)      # H/4
        self.stage3 = ConvStage(32, 64)      # H/8
        self.stage4 = ConvStage(64, 96)      # H/16
        self.projections = nn.ModuleList([nn.Conv2d(c, feature_width, 1)
                                          for c in (16, 32, 64, 96)])
        self.fuse3 = FusionBlock(feature_width)
        self.fuse2 = FusionBlock(feature_width)
        self.fuse1 = FusionBlock(feature_width)
        self.depth_head = nn.Sequential(nn.Conv2d(feature_width, 16, 3, padding=1),
                                        nn.GELU(), nn.Conv2d(16, 1, 1))

    def forward_with_shapes(self, image: Tensor) -> dict[str, Tensor]:
        """Return intermediate tensors so the executable demo exposes every stage."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected image [N,3,H,W], got {tuple(image.shape)}")
        f1 = self.stage1(image)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        p1, p2, p3, p4 = [projection(feature) for projection, feature
                          in zip(self.projections, (f1, f2, f3, f4))]
        d3 = self.fuse3(p3, p4)
        d2 = self.fuse2(p2, d3)
        d1 = self.fuse1(p1, d2)
        depth_logits = F.interpolate(self.depth_head(d1), size=image.shape[-2:],
                                     mode="bilinear", align_corners=False)
        depth = torch.sigmoid(depth_logits[:, 0]) * self.max_depth
        return {"image": image, "encoder_1": f1, "encoder_2": f2,
                "encoder_3": f3, "encoder_4": f4, "fused_3": d3,
                "fused_2": d2, "fused_1": d1, "depth_logits": depth_logits,
                "metric_depth": depth}

    def forward(self, image: Tensor) -> Tensor:
        return self.forward_with_shapes(image)["metric_depth"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyDPTDepth(max_depth=10.0).eval()
    image = torch.randn(2, 3, 64, 80)  # N=2 normalized RGB images.
    with torch.no_grad():
        trace = model.forward_with_shapes(image)
    for name, value in trace.items():
        print(f"{name:14} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("depth range (meters):", trace["metric_depth"].min().item(),
          "to", trace["metric_depth"].max().item())
