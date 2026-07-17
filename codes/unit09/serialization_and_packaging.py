"""Package model weights with architecture, labels, and preprocessing metadata.

Run: uv run --extra cpu python codes/unit09/serialization_and_packaging.py
Input `[N,3,32,32]` float32 images; output `[N,K]` float32 logits.
A state dict is safer and more portable than pickling an entire Python model,
but reconstruction still requires an explicit architecture and versioned config.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


class PackagedClassifier(nn.Module):
    """Architecture reconstructed from package metadata before loading weights."""

    def __init__(self, hidden_channels: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1), # [N,3,H,W] -> [N,F,H,W]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                     # -> [N,F,1,1]
            nn.Flatten(),
            nn.Linear(hidden_channels, num_classes),     # -> [N,K]
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images)


@dataclass(frozen=True)
class ModelMetadata:
    format_version: int
    architecture: str
    hidden_channels: int
    labels: list[str]
    input_shape: list[int]  # one image in CHW order, not including batch N
    input_dtype: str
    color_order: str
    scale: float
    mean: list[float]
    std: list[float]


def save_package(directory: Path, model: PackagedClassifier,
                 metadata: ModelMetadata) -> None:
    """Write weights and human-readable inference contract as separate files."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model_weights.pt")
    (directory / "metadata.json").write_text(
        json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8"
    )


def load_package(directory: Path) -> tuple[PackagedClassifier, ModelMetadata]:
    """Validate metadata, reconstruct architecture, then safely load tensors."""
    raw = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    metadata = ModelMetadata(**raw)
    if metadata.format_version != 1 or metadata.architecture != "PackagedClassifier":
        raise ValueError("unsupported model package format or architecture")
    if metadata.input_shape[0] != 3 or len(metadata.labels) < 2:
        raise ValueError("package must describe RGB input and at least two classes")

    model = PackagedClassifier(metadata.hidden_channels, len(metadata.labels))
    # `weights_only=True` rejects arbitrary pickled objects and loads tensor data.
    state_dict = torch.load(directory / "model_weights.pt",
                            map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.eval(), metadata


if __name__ == "__main__":
    torch.manual_seed(0)
    metadata = ModelMetadata(
        format_version=1,
        architecture="PackagedClassifier",
        hidden_channels=8,
        labels=["cat", "dog", "bird"],
        input_shape=[3, 32, 32],
        input_dtype="float32",
        color_order="RGB",
        scale=1 / 255,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    source_model = PackagedClassifier(metadata.hidden_channels, len(metadata.labels)).eval()
    images = torch.randn(2, 3, 32, 32)

    with tempfile.TemporaryDirectory(prefix="unit09_model_") as temporary:
        package_dir = Path(temporary)
        save_package(package_dir, source_model, metadata)
        loaded_model, loaded_metadata = load_package(package_dir)
        with torch.inference_mode():
            before = source_model(images)
            after = loaded_model(images)

        print(f"package files   {[path.name for path in sorted(package_dir.iterdir())]}")
        input_shape = ("N", *loaded_metadata.input_shape)
        print(f"input contract  shape={input_shape}, "
              f"dtype={loaded_metadata.input_dtype}, color={loaded_metadata.color_order}")
        print(f"images          shape={tuple(images.shape)}, dtype={images.dtype}")
        print(f"logits          shape={tuple(after.shape)}, dtype={after.dtype}")
        print(f"round-trip max difference={(before - after).abs().max().item():.1f}")
