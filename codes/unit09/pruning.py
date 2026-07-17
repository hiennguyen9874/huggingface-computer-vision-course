"""Pruning removes low-value weights or complete channels from a model.

Run: uv run --extra cpu python codes/unit09/pruning.py
Input: float32 images `[N, 3, H, W]`.
Output: float32 class logits `[N, K]` plus sparsity statistics.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.nn.utils import prune


class TinyClassifier(nn.Module):
    """Small CNN whose convolution weights are easy to inspect and prune."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # [N,3,H,W] -> [N,8,H,W]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                    # -> [N,8,1,1]
        )
        self.classifier = nn.Linear(8, num_classes)     # [N,8] -> [N,K]

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images [N,3,H,W], got {tuple(images.shape)}")
        features = self.features(images).flatten(1)     # [N,8]
        return self.classifier(features)                # [N,K]


def tensor_sparsity(tensor: Tensor) -> float:
    """Return the fraction of elements that are exactly zero in `[0, 1]`."""
    return (tensor == 0).sum().item() / tensor.numel()


def prune_unstructured(model: TinyClassifier, amount: float) -> None:
    """Zero the smallest individual weights in every Conv2d/Linear layer.

    The resulting pattern is irregular. It reduces useful information but does
    not change dense tensor shapes, so generic hardware may not become faster.
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError(f"amount must be in [0,1), got {amount}")
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  # Materialize zeros as a normal Parameter.


def prune_output_channels(conv: nn.Conv2d, amount: float) -> None:
    """Zero complete output filters of a convolution (structured pruning).

    `dim=0` groups each filter `[C_in,kH,kW]`. Real speed/size reduction needs a
    follow-up graph rewrite that removes these zero channels and matching inputs
    in the next layer; masking alone preserves the original output shape.
    """
    prune.ln_structured(conv, name="weight", amount=amount, n=2, dim=0)
    prune.remove(conv, "weight")


if __name__ == "__main__":
    torch.manual_seed(0)
    images = torch.randn(2, 3, 32, 32, dtype=torch.float32)
    baseline = TinyClassifier().eval()
    unstructured = copy.deepcopy(baseline)
    structured = copy.deepcopy(baseline)

    prune_unstructured(unstructured, amount=0.50)
    first_conv = structured.features[0]
    assert isinstance(first_conv, nn.Conv2d)
    prune_output_channels(first_conv, amount=0.50)

    with torch.no_grad():
        baseline_logits = baseline(images)
        sparse_logits = unstructured(images)
        structured_maps = structured.features(images)
        structured_logits = structured.classifier(structured_maps.flatten(1))

    zero_filters = (first_conv.weight.flatten(1).norm(dim=1) == 0).sum().item()
    print(f"images               shape={tuple(images.shape)}, dtype={images.dtype}")
    print(f"baseline logits      shape={tuple(baseline_logits.shape)}, dtype={baseline_logits.dtype}")
    print(f"unstructured logits  shape={tuple(sparse_logits.shape)}, weight sparsity="
          f"{tensor_sparsity(unstructured.features[0].weight):.0%}")
    print(f"structured maps      shape={tuple(structured_maps.shape)}, zero filters="
          f"{zero_filters}/{first_conv.out_channels}")
    print(f"structured logits    shape={tuple(structured_logits.shape)}, dtype={structured_logits.dtype}")
