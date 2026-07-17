"""MobileNet-style depthwise-separable convolutions.

Run with:
    uv run python codes/unit02/mobilenet.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DepthwiseSeparableConv(nn.Module):
    """Replace one expensive KxK convolution with two cheaper operations.

    Input:
        x: float tensor `[N, in_channels, H, W]`.
    Output:
        `[N, out_channels, H/stride, W/stride]`.

    Depthwise convolution applies one spatial filter per input channel
    (`groups=in_channels`). Pointwise 1x1 convolution then mixes channels.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)

    def forward_with_shapes(self, x: Tensor) -> dict[str, Tensor]:
        depthwise = self.depthwise(x)  # [N, Cin, H, W] -> [N, Cin, H/s, W/s]
        depthwise = F.relu(self.depthwise_bn(depthwise))
        pointwise = self.pointwise(depthwise)  # [N, Cin, h, w] -> [N, Cout, h, w]
        output = F.relu(self.pointwise_bn(pointwise))
        return {"input": x, "depthwise": depthwise, "pointwise": pointwise, "output": output}

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_shapes(x)["output"]


class MobileNetV1Educational(nn.Module):
    """A small MobileNetV1-shaped classifier for CPU demonstrations.

    Input: `[N, 3, 128, 128]` RGB images.
    Output: `[N, num_classes]` logits.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [N, 3, 128, 128] -> [N, 32, 64, 64]
        specification = [
            (32, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 512, 2),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 512, 1),
            (512, 1024, 2),
            (1024, 1024, 1),
        ]
        self.blocks = nn.ModuleList(
            DepthwiseSeparableConv(in_channels, out_channels, stride)
            for in_channels, out_channels, stride in specification
        )
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        tensors: dict[str, Tensor] = {"input": images}
        x = self.stem(images)
        tensors["stem"] = x
        for index, block in enumerate(self.blocks, start=1):
            x = block(x)
            tensors[f"depthwise_separable_{index}"] = x
        x = self.average_pool(x)  # [N, 1024, h, w] -> [N, 1024, 1, 1]
        tensors["average_pool"] = x
        x = torch.flatten(x, 1)  # -> [N, 1024]
        tensors["flatten"] = x
        tensors["logits"] = self.classifier(x)  # -> [N, num_classes]
        return tensors

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["logits"]


def standard_conv_parameters(kernel_size: int, in_channels: int, out_channels: int) -> int:
    """Parameter count excluding bias for a standard convolution."""
    return kernel_size * kernel_size * in_channels * out_channels


def separable_conv_parameters(kernel_size: int, in_channels: int, out_channels: int) -> int:
    """Depthwise parameters plus pointwise parameters, excluding bias."""
    return kernel_size * kernel_size * in_channels + in_channels * out_channels


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MobileNetV1Educational().eval()
    images = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        tensors = model.forward_with_shapes(images)
    for name, tensor in tensors.items():
        print(f"{name:28} {tuple(tensor.shape)}")

    standard = standard_conv_parameters(3, 32, 64)
    separable = separable_conv_parameters(3, 32, 64)
    print(f"standard 3x3 parameters: {standard:,}")
    print(f"separable 3x3 parameters: {separable:,}")
    print(f"parameter ratio: {standard / separable:.2f}x")
