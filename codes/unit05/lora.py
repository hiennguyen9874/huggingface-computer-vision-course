"""Low-Rank Adaptation (LoRA): freeze W and train delta_W = B @ A.

Run: uv run python codes/unit05/lora.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Linear projection with a trainable low-rank update.

    Input `[N, in_features]`; output `[N, out_features]`.
    `A` maps in_features -> rank and `B` maps rank -> out_features. B starts at
    zero, so wrapping a layer initially preserves the base layer exactly.
    """

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0) -> None:
        super().__init__()
        if not 0 < rank <= min(base.in_features, base.out_features):
            raise ValueError("rank must be positive and fit both feature dimensions")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.scale = alpha / rank

    def forward_with_shapes(self, inputs: Tensor) -> dict[str, Tensor]:
        frozen_output = self.base(inputs)  # [N, Din] -> [N, Dout]
        low_rank = F.linear(inputs, self.lora_a)  # [N, Din] -> [N, rank]
        update = F.linear(low_rank, self.lora_b) * self.scale  # -> [N, Dout]
        return {"inputs": inputs, "frozen_output": frozen_output,
                "low_rank_features": low_rank, "lora_update": update, "adapted_output": frozen_output + update}

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_with_shapes(inputs)["adapted_output"]


if __name__ == "__main__":
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(32, 48), rank=4, alpha=8)
    inputs, target = torch.randn(2, 32), torch.randn(2, 48)
    before = layer.forward_with_shapes(inputs)
    optimizer = torch.optim.SGD((layer.lora_a, layer.lora_b), lr=0.1)
    loss = F.mse_loss(before["adapted_output"], target)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    after = layer.forward_with_shapes(inputs)
    for name, value in after.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"base_trainable={layer.base.weight.requires_grad}, adapter_parameters={layer.lora_a.numel()+layer.lora_b.numel()}")
    print(f"update_norm: before={before['lora_update'].norm():.6f}, after_one_step={after['lora_update'].norm():.6f}")
