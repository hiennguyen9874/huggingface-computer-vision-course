"""Convolution and pooling mechanics used by CNNs.

Run with:
    uv run python codes/unit02/convolution_and_pooling.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def output_size(input_size: int, kernel_size: int, padding: int, stride: int) -> int:
    """Return one spatial output dimension using the CNN formula."""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def manual_convolution2d(
    image: Tensor, kernel: Tensor, padding: int = 0, stride: int = 1
) -> Tensor:
    """Apply one 2-D cross-correlation without a learnable layer.

    Input:
        image: float tensor with shape [height, width].
        kernel: float tensor with shape [kernel_height, kernel_width].
    Output:
        feature_map: float tensor with shape [output_height, output_width].

    PyTorch calls the operation in ``Conv2d`` convolution, although the
    forward pass is technically cross-correlation (the kernel is not flipped).
    """
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must both have shape [height, width]")
    if padding < 0 or stride < 1:
        raise ValueError("padding must be >= 0 and stride must be >= 1")

    padded = F.pad(image[None, None], (padding, padding, padding, padding))
    height = output_size(image.shape[0], kernel.shape[0], padding, stride)
    width = output_size(image.shape[1], kernel.shape[1], padding, stride)
    feature_map = torch.empty((height, width), dtype=image.dtype, device=image.device)

    for row in range(height):
        for column in range(width):
            row_start = row * stride
            column_start = column * stride
            window = padded[
                0,
                0,
                row_start : row_start + kernel.shape[0],
                column_start : column_start + kernel.shape[1],
            ]
            feature_map[row, column] = (window * kernel).sum()
    return feature_map


class ConvolutionAndPooling(nn.Module):
    """A tiny trainable convolution followed by ReLU and max pooling.

    Input:
        images: float tensor `[batch, 1, height, width]`.
    Output:
        pooled: float tensor `[batch, output_channels, pooled_height, pooled_width]`.
    """

    def __init__(self, output_channels: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, output_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        """Return intermediate tensors so the spatial transformations are visible."""
        convolved = self.conv(images)  # [N, 1, H, W] -> [N, 2, H, W]
        activated = F.relu(convolved)  # [N, 2, H, W] -> [N, 2, H, W]
        pooled = self.pool(activated)  # [N, 2, H, W] -> [N, 2, H/2, W/2]
        return {"input": images, "conv": convolved, "relu": activated, "pool": pooled}

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["pool"]


if __name__ == "__main__":
    torch.manual_seed(0)

    signal = torch.tensor([0.0, 1.0, 4.0, 2.0, 2.0])
    edge_kernel = torch.tensor([-1.0, 1.0])
    edge_map = F.conv1d(signal[None, None], edge_kernel[None, None])
    print(f"1-D input        {tuple(signal.shape)} -> feature map {tuple(edge_map.shape)}")
    print(f"1-D edge values: {edge_map.flatten().tolist()}")

    image = torch.arange(25.0).reshape(5, 5)
    prewitt = torch.tensor([[-1.0, 0.0, 1.0]] * 3)
    feature_map = manual_convolution2d(image, prewitt, padding=1)
    print(f"2-D image        {tuple(image.shape)} -> feature map {tuple(feature_map.shape)}")

    model = ConvolutionAndPooling()
    tensors = model.forward_with_shapes(image[None, None])
    for name, tensor in tensors.items():
        print(f"{name:16} {tuple(tensor.shape)}")
