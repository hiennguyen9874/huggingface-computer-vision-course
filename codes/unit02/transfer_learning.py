"""Transfer learning and selective fine-tuning with a ResNet-18 backbone.

Run with:
    uv run python codes/unit02/transfer_learning.py

The demo intentionally uses ``weights=None``. It illustrates the freezing and
head-replacement mechanics without downloading a checkpoint. In a real project,
pass ``ResNet18_Weights.DEFAULT`` to torchvision when pretrained weights are
available locally or network access is explicitly desired.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class TransferResNet(nn.Module):
    """ResNet-18 feature extractor with a task-specific classification head.

    Input:
        images: normalized RGB tensor `[batch, 3, height, width]`.
    Output:
        logits: float tensor `[batch, num_classes]`.
    """

    def __init__(
        self,
        num_classes: int,
        weights: ResNet18_Weights | None = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.average_pool = backbone.avgpool
        self.classifier = nn.Linear(backbone.fc.in_features, num_classes)

        if freeze_backbone:
            for parameter in self.backbone_parameters():
                parameter.requires_grad = False

    def backbone_parameters(self):
        """Yield only feature-extractor parameters, excluding the new head."""
        for module in (
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.average_pool,
        ):
            yield from module.parameters()

    def unfreeze_last_stage(self) -> None:
        """Enable fine-tuning of ResNet's highest-level feature stage."""
        for parameter in self.layer4.parameters():
            parameter.requires_grad = True

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        """Return the backbone stages and classifier output."""
        stem = self.stem(images)  # [N, 3, H, W] -> [N, 64, H/4, W/4]
        layer1 = self.layer1(stem)  # -> [N, 64, H/4, W/4]
        layer2 = self.layer2(layer1)  # -> [N, 128, H/8, W/8]
        layer3 = self.layer3(layer2)  # -> [N, 256, H/16, W/16]
        layer4 = self.layer4(layer3)  # -> [N, 512, H/32, W/32]
        pooled = self.average_pool(layer4)  # -> [N, 512, 1, 1]
        flattened = torch.flatten(pooled, 1)  # -> [N, 512]
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


def trainable_parameter_count(model: nn.Module) -> int:
    """Count parameters that an optimizer will update."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TransferResNet(num_classes=3, weights=None).eval()
    images = torch.randn(2, 3, 224, 224)  # [batch, RGB channels, height, width]

    print(f"trainable parameters (head only): {trainable_parameter_count(model):,}")
    model.unfreeze_last_stage()
    print(f"trainable parameters (+ layer4): {trainable_parameter_count(model):,}")

    with torch.no_grad():
        tensors = model.forward_with_shapes(images)
    for name, tensor in tensors.items():
        print(f"{name:14} {tuple(tensor.shape)}")
