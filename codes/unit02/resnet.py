"""Residual learning with identity and projection shortcuts.

Run with:
    uv run python codes/unit02/resnet.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BasicResidualBlock(nn.Module):
    """The ResNet-18/34 two-convolution block.

    Input:
        x: `[N, in_channels, H, W]`.
    Output:
        `[N, out_channels, H/stride, W/stride]`.

    When shape changes, the shortcut uses a learnable 1x1 projection. Otherwise
    it is identity and adds no parameters: ``output = F(x) + x``.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward_with_shapes(self, x: Tensor) -> dict[str, Tensor]:
        residual = self.shortcut(x)  # identity or [N, in, H, W] -> [N, out, H/s, W/s]
        transformed = F.relu(self.bn1(self.conv1(x)))
        transformed = self.bn2(self.conv2(transformed))
        added = transformed + residual  # same shape on both branches
        output = F.relu(added)
        return {"input": x, "residual": residual, "transformed": transformed, "output": output}

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_shapes(x)["output"]


class EducationalResNet(nn.Module):
    """A compact ResNet-18-shaped image classifier.

    Input: `[N, 3, 64, 64]` RGB images.
    Output: `[N, num_classes]` logits.

    The block counts are `(2, 2, 2, 2)` like ResNet-18, with narrower channels
    so the shape demonstration is inexpensive on CPU.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # [N, 3, 64, 64] -> [N, 32, 32, 32]
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)  # -> 32x32
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)  # -> 16x16
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)  # -> 8x8
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=2)  # -> 4x4
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [BasicResidualBlock(in_channels, out_channels, stride)]
        layers.extend(BasicResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        stem = self.stem(images)
        layer1 = self.layer1(stem)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        pooled = self.average_pool(layer4)  # [N, 256, 4, 4] -> [N, 256, 1, 1]
        flattened = torch.flatten(pooled, 1)  # -> [N, 256]
        logits = self.classifier(flattened)  # -> [N, num_classes]
        return {
            "input": images,
            "stem": stem,
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
            "average_pool": pooled,
            "flatten": flattened,
            "logits": logits,
        }

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = EducationalResNet().eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        tensors = model.forward_with_shapes(images)
    for name, tensor in tensors.items():
        print(f"{name:14} {tuple(tensor.shape)}")

    identity_block = BasicResidualBlock(32, 32)
    projection_block = BasicResidualBlock(32, 64, stride=2)
    example = torch.randn(1, 32, 16, 16)
    print(f"identity shortcut:   {tuple(identity_block(example).shape)}")
    print(f"projection shortcut: {tuple(projection_block(example).shape)}")
