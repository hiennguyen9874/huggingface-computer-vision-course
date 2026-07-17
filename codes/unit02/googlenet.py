"""Compact GoogLeNet/Inception implementation for shape exploration.

Run with:
    uv run python codes/unit02/googlenet.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConvBNReLU(nn.Module):
    """A convolution branch used by Inception.

    Input/output: `[N, channels, height, width]`; padding preserves spatial size
    when stride is one.
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class InceptionModule(nn.Module):
    """Four parallel receptive-field branches concatenated by channel.

    Input:
        x: `[N, in_channels, H, W]`.
    Output:
        `[N, n1x1 + n3x3 + n5x5 + pool_proj, H, W]`.

    The 1x1 convolutions before 3x3/5x5 are bottlenecks: they reduce channel
    count before the expensive spatial convolution.
    """

    def __init__(
        self,
        in_channels: int,
        n1x1: int,
        n3x3_reduce: int,
        n3x3: int,
        n5x5_reduce: int,
        n5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        self.branch1 = ConvBNReLU(in_channels, n1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, n3x3_reduce, kernel_size=1),
            ConvBNReLU(n3x3_reduce, n3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, n5x5_reduce, kernel_size=1),
            ConvBNReLU(n5x5_reduce, n5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, pool_proj, kernel_size=1),
        )

    def forward_with_branches(self, x: Tensor) -> dict[str, Tensor]:
        branches = {
            "1x1": self.branch1(x),
            "3x3": self.branch2(x),
            "5x5": self.branch3(x),
            "pool+1x1": self.branch4(x),
        }
        branches["concat"] = torch.cat(list(branches.values()), dim=1)
        return branches

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_branches(x)["concat"]


class AuxiliaryClassifier(nn.Module):
    """A small mid-network classifier used only to aid training.

    Input: `[N, in_channels, H, W]`.
    Output: `[N, num_classes]`.
    Adaptive pooling makes this educational version work at several input sizes.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)  # [N, C, H, W] -> [N, C, 1, 1]
        x = F.relu(self.projection(x))  # -> [N, 64, 1, 1]
        return self.classifier(torch.flatten(x, 1))  # -> [N, num_classes]


class CompactGoogLeNet(nn.Module):
    """A compact Inception network retaining GoogLeNet's key ideas.

    Input: `[N, 3, 128, 128]` RGB images.
    Output: main logits `[N, num_classes]`; optional auxiliary logits `[N, num_classes]`.
    """

    def __init__(self, num_classes: int = 10, use_auxiliary: bool = True) -> None:
        super().__init__()
        self.use_auxiliary = use_auxiliary
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.MaxPool2d(2),  # 64 -> 32
            ConvBNReLU(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),  # 32 -> 16
        )
        self.inception1 = InceptionModule(64, 16, 16, 24, 4, 8, 8)  # -> 56 channels
        self.inception2 = InceptionModule(56, 24, 16, 32, 8, 12, 12)  # -> 80 channels
        self.downsample = nn.MaxPool2d(2)  # 16 -> 8
        self.inception3 = InceptionModule(80, 32, 24, 48, 8, 16, 16)  # -> 112 channels
        self.inception4 = InceptionModule(112, 48, 32, 64, 8, 16, 16)  # -> 144 channels
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(144, num_classes)
        self.auxiliary = AuxiliaryClassifier(112, num_classes)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        stem = self.stem(images)
        inception1 = self.inception1(stem)
        inception2 = self.inception2(inception1)
        downsampled = self.downsample(inception2)
        inception3 = self.inception3(downsampled)
        auxiliary = self.auxiliary(inception3) if self.use_auxiliary else None
        inception4 = self.inception4(inception3)
        pooled = self.average_pool(inception4)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        result: dict[str, Tensor] = {
            "input": images,
            "stem": stem,
            "inception1": inception1,
            "inception2": inception2,
            "downsample": downsampled,
            "inception3": inception3,
            "inception4": inception4,
            "average_pool": pooled,
            "flatten": flattened,
            "logits": logits,
        }
        if auxiliary is not None:
            result["auxiliary_logits"] = auxiliary
        return result

    def forward(self, images: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        tensors = self.forward_with_shapes(images)
        if self.training and self.use_auxiliary:
            return tensors["logits"], tensors["auxiliary_logits"]
        return tensors["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CompactGoogLeNet(num_classes=10, use_auxiliary=True).eval()
    images = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        tensors = model.forward_with_shapes(images)
    for name, tensor in tensors.items():
        print(f"{name:18} {tuple(tensor.shape)}")

    # Each branch sees the same spatial dimensions and concatenation increases channels.
    branches = model.inception1.forward_with_branches(tensors["stem"])
    print("Inception1 branches:")
    for name, tensor in branches.items():
        print(f"  {name:12} {tuple(tensor.shape)}")
