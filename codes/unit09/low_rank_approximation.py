"""Low-rank approximation replaces one large Linear weight with two factors.

Run: uv run --extra cpu python codes/unit09/low_rank_approximation.py
Input `[N,I]` -> first factor `[N,R]` -> output `[N,O]`, where R < I,O.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LowRankLinear(nn.Module):
    """Two Linear layers initialized from a truncated SVD of a dense layer."""

    def __init__(self, down: nn.Linear, up: nn.Linear) -> None:
        super().__init__()
        self.down = down  # [N,I] -> [N,R], weight [R,I]
        self.up = up      # [N,R] -> [N,O], weight [O,R]

    @classmethod
    def from_linear(cls, source: nn.Linear, rank: int) -> "LowRankLinear":
        """Approximate `W[O,I]` as `(U*S)[O,R] @ Vh[R,I]`."""
        output_dim, input_dim = source.weight.shape
        if not 1 <= rank <= min(input_dim, output_dim):
            raise ValueError(f"rank must be in [1,{min(input_dim, output_dim)}]")
        with torch.no_grad():
            u, singular_values, vh = torch.linalg.svd(source.weight, full_matrices=False)
            down = nn.Linear(input_dim, rank, bias=False)
            up = nn.Linear(rank, output_dim, bias=source.bias is not None)
            down.weight.copy_(vh[:rank])
            up.weight.copy_(u[:, :rank] * singular_values[:rank])
            if source.bias is not None:
                up.bias.copy_(source.bias)
        return cls(down, up)

    def forward_with_shapes(self, inputs: Tensor) -> dict[str, Tensor]:
        compressed = self.down(inputs)
        outputs = self.up(compressed)
        return {"inputs": inputs, "rank_features": compressed, "outputs": outputs}

    def forward(self, inputs: Tensor) -> Tensor:
        return self.up(self.down(inputs))


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    dense = nn.Linear(128, 64)          # weight [O=64,I=128]
    low_rank = LowRankLinear.from_linear(dense, rank=16)
    inputs = torch.randn(8, 128)        # [N=8,I=128]
    with torch.no_grad():
        dense_outputs = dense(inputs)   # [N,O]
        steps = low_rank.forward_with_shapes(inputs)

    for name, tensor in steps.items():
        print(f"{name:14} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    relative_error = (dense_outputs - steps["outputs"]).norm() / dense_outputs.norm()
    print(f"dense weight   shape={tuple(dense.weight.shape)}")
    print(f"factor weights shape={tuple(low_rank.down.weight.shape)} x "
          f"{tuple(low_rank.up.weight.shape)}")
    print(f"parameters: dense={parameter_count(dense):,}, low-rank={parameter_count(low_rank):,}")
    print(f"relative output error={relative_error.item():.4f}")
