"""The basic CNN from the Unit 2 introduction.

Run with:
    uv run python codes/unit02/basic_cnn.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BasicCNN(nn.Module):
    """A 28x28 grayscale image classifier.

    Input:
        images: float tensor `[batch, 1, 28, 28]`.
    Output:
        logits: float tensor `[batch, num_classes]`.

    The convolution layers intentionally use no padding, matching the chapter:
    28 -> 26 -> 13 -> 11 -> 5 in the spatial dimensions.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # [N, 1, 28, 28] -> [N, 32, 26, 26]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # [N, 32, 13, 13] -> [N, 64, 11, 11]
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(64 * 5 * 5, num_classes)  # [N, 1600] -> [N, 10]

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        """Compute and return every major intermediate tensor."""
        first_conv = self.conv1(images)
        first_relu = F.relu(first_conv)
        first_pool = F.max_pool2d(first_relu, kernel_size=2)  # 26 -> 13

        second_conv = self.conv2(first_pool)
        second_relu = F.relu(second_conv)
        second_pool = F.max_pool2d(second_relu, kernel_size=2)  # 11 -> 5

        flattened = torch.flatten(second_pool, start_dim=1)  # [N, 64, 5, 5] -> [N, 1600]
        dropped = self.dropout(flattened)
        logits = self.classifier(dropped)  # [N, 1600] -> [N, num_classes]
        return {
            "input": images,
            "conv1": first_conv,
            "relu1": first_relu,
            "pool1": first_pool,
            "conv2": second_conv,
            "relu2": second_relu,
            "pool2": second_pool,
            "flatten": flattened,
            "dropout": dropped,
            "logits": logits,
        }

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BasicCNN().eval()  # eval makes the demo's dropout output deterministic.
    images = torch.randn(2, 1, 28, 28)  # [batch=2, channels=1, height=28, width=28]

    with torch.no_grad():
        tensors = model.forward_with_shapes(images)
    for name, tensor in tensors.items():
        print(f"{name:10} {tuple(tensor.shape)}")
