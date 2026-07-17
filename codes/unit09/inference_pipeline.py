"""Complete CV request path: preprocess -> model -> postprocess response.

Run: uv run --extra cpu python codes/unit09/inference_pipeline.py
External input is one uint8 RGB image `[H,W,3]`; model input is normalized
float32 `[1,3,32,32]`; model output is logits `[1,K]`; the public output is a
JSON-ready label/confidence object. Training and serving preprocessing must match.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),   # -> [N,8,H,W]
            nn.AdaptiveAvgPool2d(1),                    # -> [N,8,1,1]
        )
        self.head = nn.Linear(8, num_classes)           # [N,8] -> [N,K]

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.features(images).flatten(1))


@dataclass(frozen=True)
class Prediction:
    class_id: int
    label: str
    confidence: float


def preprocess(rgb_image: Tensor, size: tuple[int, int] = (32, 32)) -> dict[str, Tensor]:
    """Convert uint8 HWC RGB pixels into normalized float32 NCHW input."""
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3 or rgb_image.dtype != torch.uint8:
        raise ValueError(f"expected uint8 RGB image [H,W,3], got "
                         f"shape={tuple(rgb_image.shape)}, dtype={rgb_image.dtype}")
    nchw = rgb_image.permute(2, 0, 1).unsqueeze(0).float() # [1,3,H,W]
    scaled = nchw / 255.0                                 # float32 values [0,1]
    resized = F.interpolate(scaled, size=size, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (resized - mean) / std                   # [1,3,32,32]
    return {"raw_hwc": rgb_image, "nchw": nchw, "scaled": scaled,
            "resized": resized, "normalized": normalized}


def postprocess(logits: Tensor, labels: list[str]) -> Prediction:
    """Convert one `[1,K]` logits tensor to a stable external response."""
    if logits.shape != (1, len(labels)):
        raise ValueError(f"expected logits [1,{len(labels)}], got {tuple(logits.shape)}")
    probabilities = logits.softmax(dim=1)                 # [1,K], rows sum to 1
    confidence, class_id = probabilities.max(dim=1)       # each has shape [1]
    index = class_id.item()
    return Prediction(index, labels[index], confidence.item())


def predict(model: nn.Module, rgb_image: Tensor, labels: list[str]) -> Prediction:
    """Serving boundary used by an HTTP/gRPC handler after decoding image bytes."""
    model_input = preprocess(rgb_image)["normalized"]
    with torch.inference_mode():
        logits = model(model_input)
    return postprocess(logits, labels)


if __name__ == "__main__":
    torch.manual_seed(0)
    labels = ["cat", "dog", "bird"]
    model = ImageClassifier(len(labels)).eval()
    raw_image = torch.randint(0, 256, (48, 64, 3), dtype=torch.uint8)
    steps = preprocess(raw_image)
    with torch.inference_mode():
        logits = model(steps["normalized"])
    response = postprocess(logits, labels)

    for name, tensor in steps.items():
        value_range = (tensor.min().item(), tensor.max().item())
        print(f"{name:10} shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
              f"range=({value_range[0]:.3f},{value_range[1]:.3f})")
    print(f"logits     shape={tuple(logits.shape)}, dtype={logits.dtype}")
    print(f"response   {asdict(response)}")
