"""Post-training dynamic quantization stores selected weights as INT8.

Run: uv run --extra cpu python codes/unit09/quantization.py
Input: float32 feature batches `[N, F]`; output: float32 logits `[N, K]`.
Dynamic quantization quantizes Linear weights ahead of time and activations at
runtime. It is a compact CPU example; CNN deployment often uses static PTQ/QAT.
"""

from __future__ import annotations

import io

import torch
from torch import Tensor, nn


class FeatureClassifier(nn.Module):
    """MLP classification head representative of a CV model's final layers."""

    def __init__(self, feature_dim: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 128),  # [N,F] -> [N,128]
            nn.ReLU(),
            nn.Linear(128, num_classes),  # [N,128] -> [N,K]
        )

    def forward_with_shapes(self, features: Tensor) -> dict[str, Tensor]:
        if features.ndim != 2 or features.shape[1] != self.layers[0].in_features:
            expected = self.layers[0].in_features
            raise ValueError(f"expected features [N,{expected}], got {tuple(features.shape)}")
        hidden = self.layers[0](features)       # float32 [N,128]
        activated = self.layers[1](hidden)      # float32 [N,128]
        logits = self.layers[2](activated)      # float32 [N,K]
        return {"features": features, "hidden": hidden,
                "activated": activated, "logits": logits}

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


def serialized_size_bytes(model: nn.Module) -> int:
    """Measure state-dict bytes; useful for comparing deployable artifacts."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


if __name__ == "__main__":
    torch.manual_seed(0)
    fp32_model = FeatureClassifier().eval()
    features = torch.randn(4, 256, dtype=torch.float32)

    # `torch.ao.quantization` is the stable namespace in this PyTorch version.
    # Only Linear modules are converted; ReLU remains a regular float operator.
    int8_model = torch.ao.quantization.quantize_dynamic(
        fp32_model, {nn.Linear}, dtype=torch.qint8
    )
    with torch.no_grad():
        fp32_steps = fp32_model.forward_with_shapes(features)
        int8_logits = int8_model(features)

    for name, tensor in fp32_steps.items():
        print(f"{name:12} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    first_quantized = int8_model.layers[0]
    quantized_weight = first_quantized.weight()
    print(f"INT8 weight  shape={tuple(quantized_weight.shape)}, dtype={quantized_weight.dtype}")
    print(f"INT8 logits  shape={tuple(int8_logits.shape)}, dtype={int8_logits.dtype}")
    print(f"artifact size: FP32={serialized_size_bytes(fp32_model):,} bytes, "
          f"INT8={serialized_size_bytes(int8_model):,} bytes")
    print(f"max output difference={(fp32_steps['logits'] - int8_logits).abs().max().item():.6f}")
