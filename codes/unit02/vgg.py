"""A teaching-scale VGG-16 classifier.

Run with:
    uv run python codes/unit02/vgg.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VGG16(nn.Module):
    """VGG's five blocks of 3x3 convolutions and 2x2 max pooling.

    Input:
        images: float RGB tensor `[batch, 3, 224, 224]`.
    Output:
        logits: float tensor `[batch, num_classes]`.

    The original VGG classifier uses two 4096-unit layers and is very large.
    ``classifier_hidden`` keeps this executable lesson small while preserving
    the convolutional architecture and its `[N, 512, 7, 7]` feature shape.
    """

    def __init__(self, num_classes: int = 10, classifier_hidden: int = 256) -> None:
        super().__init__()
        self.block1 = self._block(3, 64, 2)  # [N, 3, 224, 224] -> [N, 64, 112, 112]
        self.block2 = self._block(64, 128, 2)  # -> [N, 128, 56, 56]
        self.block3 = self._block(128, 256, 3)  # -> [N, 256, 28, 28]
        self.block4 = self._block(256, 512, 3)  # -> [N, 512, 14, 14]
        self.block5 = self._block(512, 512, 3)  # -> [N, 512, 7, 7]
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, classifier_hidden),  # [N, 25088] -> [N, hidden]
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(classifier_hidden, num_classes),  # [N, hidden] -> [N, classes]
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int, convolution_count: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for layer_index in range(convolution_count):
            input_channels = in_channels if layer_index == 0 else out_channels
            layers.extend(
                [
                    nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        """Return the output of each VGG block and classifier step."""
        block1 = self.block1(images)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        flattened = torch.flatten(block5, 1)  # [N, 512, 7, 7] -> [N, 25088]
        hidden = F.relu(self.classifier[0](flattened))
        dropped = self.classifier[2](hidden)
        logits = self.classifier[3](dropped)
        return {
            "input": images,
            "block1": block1,
            "block2": block2,
            "block3": block3,
            "block4": block4,
            "block5": block5,
            "flatten": flattened,
            "hidden": hidden,
            "dropout": dropped,
            "logits": logits,
        }

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = VGG16(num_classes=10).eval()
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        tensors = model.forward_with_shapes(images)
    for name, tensor in tensors.items():
        print(f"{name:10} {tuple(tensor.shape)}")
