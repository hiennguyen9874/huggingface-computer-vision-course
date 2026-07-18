"""Multi-scale retention in equivalent parallel and recurrent forms.

The same causal weighted key/value sum can train in parallel and infer one token
at a time with a fixed-size state. This real-valued teaching implementation
omits RetNet's rotary/complex positional details and normalization.

Run: uv run --extra cpu python codes/unit13/retention.py
Notation: N=batch, L=sequence, D=model width, M=heads, Dh=D/M.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiScaleRetention(nn.Module):
    """Causal retention with one learned decay per head.

    Input/output: `[N,L,D]`. Parallel mode materializes `[M,L,L]` for clarity;
    recurrent mode keeps state `[N,M,Dh,Dh]`, independent of sequence length.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        # sigmoid(logit) guarantees a stable decay in (0,1), initialized at
        # multiple time scales: some heads forget faster than others.
        initial_decays = torch.linspace(0.80, 0.98, num_heads)
        self.decay_logits = nn.Parameter(torch.logit(initial_decays))

    def _project(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError(f"expected [N,L,{self.embed_dim}], got {tuple(tokens.shape)}")
        batch, length, _ = tokens.shape
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)                         # each [N,L,D]
        reshape = lambda x: x.view(batch, length, self.num_heads,
                                   self.head_dim).transpose(1, 2)           # [N,M,L,Dh]
        return reshape(q), reshape(k), reshape(v)

    def _merge_heads(self, heads: Tensor) -> Tensor:
        batch, _, length, _ = heads.shape
        return heads.transpose(1, 2).reshape(batch, length, self.embed_dim) # [N,L,D]

    def parallel(self, tokens: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Evaluate all positions together; useful for GPU training."""
        q, k, v = self._project(tokens)                                     # [N,M,L,Dh]
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        age = positions[:, None] - positions[None, :]                       # [L,L]: i-j
        causal = age >= 0
        decays = torch.sigmoid(self.decay_logits).to(tokens.dtype)          # [M]
        decay_mask = decays[:, None, None] ** age.clamp_min(0)[None]        # [M,L,L]
        decay_mask = decay_mask * causal[None]                              # future entries = 0
        similarities = torch.einsum("nmld,nmsd->nmls", q, k)               # [N,M,L,L]
        retained_heads = torch.einsum(
            "nmls,mls,nmse->nmle", similarities, decay_mask, v
        ) / self.head_dim ** 0.5                                            # [N,M,L,Dh]
        merged = self._merge_heads(retained_heads)                          # [N,L,D]
        output = self.output_projection(merged)                             # [N,L,D]
        return output, {"query_heads": q, "key_heads": k, "value_heads": v,
                        "decay_mask": decay_mask, "retained_heads": retained_heads,
                        "output": output}

    def recurrent(self, tokens: Tensor, state: Tensor | None = None
                  ) -> tuple[Tensor, Tensor]:
        """Evaluate left-to-right with constant-size state.

        Args:
            tokens: `[N,L,D]`; L may be one for streaming inference.
            state: optional previous state `[N,M,Dh,Dh]`.
        Returns:
            outputs `[N,L,D]`, next state `[N,M,Dh,Dh]`.
        """
        q, k, v = self._project(tokens)
        batch = tokens.shape[0]
        expected = (batch, self.num_heads, self.head_dim, self.head_dim)
        if state is None:
            state = tokens.new_zeros(expected)
        elif state.shape != expected:
            raise ValueError(f"expected state {expected}, got {tuple(state.shape)}")
        decay = torch.sigmoid(self.decay_logits).to(tokens.dtype)[None, :, None, None]
        output_steps: list[Tensor] = []
        for index in range(tokens.shape[1]):
            # Outer product k_i^T v_i contributes one rank-1 memory update.
            kv = torch.einsum("nmd,nme->nmde", k[:, :, index], v[:, :, index])
            state = decay * state + kv                                     # [N,M,Dh,Dh]
            step = torch.einsum("nmd,nmde->nme", q[:, :, index], state)
            output_steps.append(step / self.head_dim ** 0.5)                # [N,M,Dh]
        heads = torch.stack(output_steps, dim=2)                            # [N,M,L,Dh]
        return self.output_projection(self._merge_heads(heads)), state

    def forward(self, tokens: Tensor, mode: str = "parallel") -> Tensor:
        if mode == "parallel":
            return self.parallel(tokens)[0]
        if mode == "recurrent":
            return self.recurrent(tokens)[0]
        raise ValueError(f"mode must be 'parallel' or 'recurrent', got {mode!r}")


if __name__ == "__main__":
    torch.manual_seed(0)
    layer = MultiScaleRetention(embed_dim=32, num_heads=4).eval()
    tokens = torch.randn(2, 12, 32)
    with torch.no_grad():
        parallel_output, trace = layer.parallel(tokens)
        recurrent_output, final_state = layer.recurrent(tokens)
    for name, value in {"tokens": tokens, **trace,
                        "recurrent_output": recurrent_output,
                        "final_recurrent_state": final_state}.items():
        print(f"{name:23} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("max |parallel - recurrent| =",
          f"{(parallel_output - recurrent_output).abs().max().item():.2e}")
