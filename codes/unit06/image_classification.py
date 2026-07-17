"""Image classification: one label distribution for each whole image.

Run: uv run python codes/unit06/image_classification.py
Inputs are float RGB tensors `[N, 3, H, W]`; outputs are class logits `[N, K]`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ImageClassifier(nn.Module):
    """A small CNN classifier for RGB images whose height/width are divisible by 4."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [N,3,H,W] -> [N,32,H,W]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [N,32,H/2,W/2]
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> [N,64,H/2,W/2]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [N,64,H/4,W/4]
        )
        # Global pooling makes the classifier accept different spatial sizes.
        self.pool = nn.AdaptiveAvgPool2d(1)              # -> [N,64,1,1]
        self.classifier = nn.Linear(64, num_classes)     # [N,64] -> [N,K]

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        """Return each major step so the tensor contract is executable."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected RGB images [N, 3, H, W], got {tuple(images.shape)}")
        feature_maps = self.features(images)
        image_features = self.pool(feature_maps).flatten(1)
        logits = self.classifier(image_features)
        probabilities = logits.softmax(dim=-1)
        predicted_class = probabilities.argmax(dim=-1)  # [N], integer class index
        return {
            "images": images,
            "feature_maps": feature_maps,
            "image_features": image_features,
            "logits": logits,
            "probabilities": probabilities,
            "predicted_class": predicted_class,
        }

    def forward(self, images: Tensor) -> Tensor:
        """Return unnormalized class scores `[N, K]` for cross-entropy training."""
        return self.classifier(self.pool(self.features(images)).flatten(1))


if __name__ == "__main__":
    torch.manual_seed(0)
    class_names = ["cat", "dog", "horse"]
    model = ImageClassifier(num_classes=len(class_names)).eval()
    images = torch.rand(2, 3, 64, 64, dtype=torch.float32)  # values conventionally in [0,1]
    with torch.no_grad():
        steps = model.forward_with_shapes(images)
    for name, value in steps.items():
        print(f"{name:18} shape={tuple(value.shape)}, dtype={value.dtype}")
    for index, class_id in enumerate(steps["predicted_class"].tolist()):
        score = steps["probabilities"][index, class_id].item()
        print(f"image {index}: label={class_names[class_id]!r}, confidence={score:.3f}")
