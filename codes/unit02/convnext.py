"""A compact ConvNeXt-style CNN for the 2020s.

Run with:
    uv run python codes/unit02/convnext.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """Apply LayerNorm over channels while preserving NCHW at the boundary.

    Input/output: `[N, C, H, W]`. LayerNorm itself normalizes the last axis,
    so the tensor temporarily becomes `[N, H, W, C]`.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.normalization(x)
        return x.permute(0, 3, 1, 2).contiguous()  # -> [N, C, H, W]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt's residual block: spatial mixing, then channel mixing.

    Input/output: `[N, channels, H, W]`.
    The depthwise 7x7 convolution sees a large local neighborhood. The two
    linear layers are equivalent to 1x1 convolutions and expand channels 4x.
    """

    def __init__(self, channels: int, layer_scale: float = 1e-6) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )  # [N, C, H, W] -> [N, C, H, W]
        self.normalization = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, 4 * channels)  # [N,H,W,C] -> [N,H,W,4C]
        self.project = nn.Linear(4 * channels, channels)  # [N,H,W,4C] -> [N,H,W,C]
        self.layer_scale = nn.Parameter(layer_scale * torch.ones(channels))

    def forward_with_shapes(self, x: Tensor) -> dict[str, Tensor]:
        residual = x
        depthwise = self.depthwise(x)
        channels_last = depthwise.permute(0, 2, 3, 1)
        normalized = self.normalization(channels_last)
        expanded = self.expand(normalized)
        activated = torch.nn.functional.gelu(expanded)
        projected = self.project(activated)
        scaled = projected * self.layer_scale
        output = scaled.permute(0, 3, 1, 2).contiguous() + residual
        return {
            "input": x,
            "depthwise": depthwise,
            "normalized_nhwc": normalized,
            "expanded_nhwc": expanded,
            "projected_nhwc": projected,
            "output": output,
        }

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_shapes(x)["output"]


class CompactConvNeXt(nn.Module):
    """A scaled-down ConvNeXt with the chapter's defining design choices.

    Input: `[N, 3, 128, 128]` RGB images.
    Output: `[N, num_classes]` logits.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        dimensions = (32, 64, 128, 256)
        depths = (1, 1, 2, 1)
        self.patchify = nn.Sequential(
            nn.Conv2d(3, dimensions[0], kernel_size=4, stride=4),
            LayerNorm2d(dimensions[0]),
        )  # [N, 3, 128, 128] -> [N, 32, 32, 32]
        self.stages = nn.ModuleList(
            nn.Sequential(*(ConvNeXtBlock(dimensions[index]) for _ in range(depths[index])))
            for index in range(4)
        )
        self.downsamples = nn.ModuleList(
            nn.Sequential(LayerNorm2d(dimensions[index]), nn.Conv2d(dimensions[index], dimensions[index + 1], 2, 2))
            for index in range(3)
        )
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dimensions[-1], num_classes)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        tensors: dict[str, Tensor] = {"input": images}
        x = self.patchify(images)
        tensors["patchify_stem"] = x
        for index, stage in enumerate(self.stages):
            x = stage(x)
            tensors[f"stage{index + 1}"] = x
            if index < len(self.downsamples):
                x = self.downsamples[index](x)
                tensors[f"downsample{index + 1}"] = x
        x = self.average_pool(x)  # [N, 256, H, W] -> [N, 256, 1, 1]
        tensors["average_pool"] = x
        x = torch.flatten(x, 1)  # -> [N, 256]
        tensors["flatten"] = x
        tensors["logits"] = self.classifier(x)  # -> [N, num_classes]
        return tensors

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CompactConvNeXt().eval()
    images = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        tensors = model.forward_with_shapes(images)
    for name, tensor in tensors.items():
        print(f"{name:18} {tuple(tensor.shape)}")

    block_tensors = model.stages[0][0].forward_with_shapes(tensors["patchify_stem"])
    print("first ConvNeXt block internals:")
    for name, tensor in block_tensors.items():
        print(f"  {name:16} {tuple(tensor.shape)}")
