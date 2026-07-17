"""Zero-shot, linear probing, and selective fine-tuning for a VLM.

Run with:
    uv run python codes/unit04/transfer_learning.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from clip import EducationalCLIP


class CLIPLinearProbe(nn.Module):
    """Freeze a CLIP-style model and learn a task-specific image classifier.

    Input: RGB images `[batch, 3, 32, 32]`.
    Output: task logits `[batch, number_of_classes]`.
    """

    def __init__(self, backbone: EducationalCLIP, embedding_dim: int, classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embedding_dim, classes)
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Linear-probe mode: update only the new classifier."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_image_projection(self) -> None:
        """Fine-tuning mode: also adapt the final visual projection layer."""
        for parameter in self.backbone.image_encoder.projection.parameters():
            parameter.requires_grad = True

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        features = self.backbone.image_encoder(images)  # [N, 3, H, W] -> [N, D]
        logits = self.classifier(features)  # [N, D] -> [N, classes]
        return {"images": images, "image_features": features, "task_logits": logits}

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["task_logits"]


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the number of scalar parameters an optimizer can update."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    clip = EducationalCLIP(vocabulary_size=100, embedding_dim=32)

    # Zero-shot uses prompt similarities and changes no parameters.
    images = torch.randn(2, 3, 32, 32)
    prompts = torch.randint(0, 100, (4, 6), dtype=torch.long)
    with torch.no_grad():
        zero_shot_logits = clip(images, prompts)  # [2 images, 4 prompt classes]
    print(f"zero-shot logits      shape={tuple(zero_shot_logits.shape)}")

    # Linear probing trains a small head on labeled target-domain examples.
    probe = CLIPLinearProbe(clip, embedding_dim=32, classes=3).eval()
    print(f"linear-probe params   {trainable_parameter_count(probe):,}")
    with torch.no_grad():
        steps = probe.forward_with_shapes(images)
    for name, tensor in steps.items():
        print(f"{name:21} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    # Selective fine-tuning uses a small learning rate for a limited backbone part.
    probe.unfreeze_image_projection()
    print(f"selective-tune params {trainable_parameter_count(probe):,}")
    optimizer_groups = [
        {"params": probe.classifier.parameters(), "lr": 1e-3},
        {"params": probe.backbone.image_encoder.projection.parameters(), "lr": 1e-5},
    ]
    optimizer = torch.optim.SGD(optimizer_groups)
    print("optimizer learning rates:", [group["lr"] for group in optimizer.param_groups])
