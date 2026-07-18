"""Hyena: implicit long convolution and input-dependent gating.

This is a small second-order Hyena-style image classifier, not the optimized
research implementation. It preserves the defining data flow while using only
PyTorch operations that run on CPU.

Run: uv run --extra cpu python codes/unit13/hyena.py
Notation: N=batch, L=patch tokens, D=model width, P=patch size.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ImplicitLongFilter(nn.Module):
    """Generate a length-L depthwise kernel instead of storing L parameters.

    Input: integer L. Output: kernel `[L,D]`, one value per position/channel.
    The exponential envelope makes distant coefficients stable at initialization.
    """

    def __init__(self, channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.channels = channels
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, channels),
        )
        self.log_decay = nn.Parameter(torch.zeros(channels))

    def forward(self, length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if length < 1:
            raise ValueError(f"length must be positive, got {length}")
        t = torch.linspace(0, 1, length, device=device, dtype=dtype)[:, None]  # [L,1]
        position_features = torch.cat(
            (t, torch.sin(2 * torch.pi * t), torch.cos(2 * torch.pi * t)), dim=-1
        )  # [L,3]
        raw_kernel = self.network(position_features)                         # [L,D]
        decay = torch.exp(-t * torch.nn.functional.softplus(self.log_decay)) # [L,D]
        return raw_kernel * decay                                            # [L,D]


def fft_long_convolution(sequence: Tensor, kernel: Tensor) -> Tensor:
    """Apply causal depthwise convolution in O(L log L).

    Args:
        sequence: float tensor `[N,L,D]`.
        kernel: float tensor `[L,D]`; channel d only filters channel d.
    Returns:
        Tensor `[N,L,D]`; output i depends on input positions `0..i`.
    """
    if sequence.ndim != 3 or kernel.shape != sequence.shape[1:]:
        raise ValueError(
            f"expected sequence [N,L,D] and kernel [L,D], got "
            f"{tuple(sequence.shape)} and {tuple(kernel.shape)}"
        )
    length = sequence.shape[1]
    fft_size = 2 * length  # zero-padding prevents circular wrap-around
    sequence_fft = torch.fft.rfft(sequence.transpose(1, 2), n=fft_size)  # [N,D,F]
    kernel_fft = torch.fft.rfft(kernel.transpose(0, 1), n=fft_size)      # [D,F]
    convolved = torch.fft.irfft(sequence_fft * kernel_fft[None], n=fft_size)
    return convolved[..., :length].transpose(1, 2)                       # [N,L,D]


class HyenaOperator(nn.Module):
    """Second-order Hyena-style token mixer: projection -> conv/gate twice.

    Input/output: `[N,L,D]`. Unlike attention, no `[N,L,L]` matrix is formed.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_projection = nn.Linear(embed_dim, 3 * embed_dim)
        # Short depthwise convolution injects local ordering into each projection.
        self.short_filter = nn.Conv1d(
            3 * embed_dim, 3 * embed_dim, kernel_size=3, padding=1,
            groups=3 * embed_dim,
        )
        self.filters = nn.ModuleList([ImplicitLongFilter(embed_dim) for _ in range(2)])
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward_with_shapes(self, tokens: Tensor) -> dict[str, Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.embed_dim:
            raise ValueError(f"expected [N,L,{self.embed_dim}], got {tuple(tokens.shape)}")
        projected = self.input_projection(tokens)                          # [N,L,3D]
        projected = self.short_filter(projected.transpose(1, 2)).transpose(1, 2)
        gate_1, gate_2, value = projected.chunk(3, dim=-1)                 # each [N,L,D]
        kernels = [module(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)
                   for module in self.filters]                             # 2 x [L,D]
        mixed_1 = fft_long_convolution(value, kernels[0])                  # [N,L,D]
        gated_1 = torch.sigmoid(gate_2) * mixed_1                          # [N,L,D]
        mixed_2 = fft_long_convolution(gated_1, kernels[1])                # [N,L,D]
        gated_2 = torch.sigmoid(gate_1) * mixed_2                          # [N,L,D]
        output = self.output_projection(gated_2)                           # [N,L,D]
        return {"tokens": tokens, "projected": projected, "kernel_1": kernels[0],
                "mixed_1": mixed_1, "gated_1": gated_1, "mixed_2": mixed_2,
                "gated_2": gated_2, "output": output}

    def forward(self, tokens: Tensor) -> Tensor:
        return self.forward_with_shapes(tokens)["output"]


class TinyHyenaVision(nn.Module):
    """Map RGB images `[N,3,H,W]` to class logits `[N,K]`."""

    def __init__(self, patch_size: int = 8, embed_dim: int = 48,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.mixer = HyenaOperator(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
                                 nn.Linear(4 * embed_dim, embed_dim))
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected RGB images [N,3,H,W], got {tuple(images.shape)}")
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size")
        patch_map = self.patch_embedding(images)                            # [N,D,H/P,W/P]
        tokens = patch_map.flatten(2).transpose(1, 2)                       # [N,L,D]
        mixed = tokens + self.mixer(self.norm_1(tokens))                    # [N,L,D]
        encoded = mixed + self.mlp(self.norm_2(mixed))                      # [N,L,D]
        pooled = encoded.mean(dim=1)                                        # [N,D]
        logits = self.head(pooled)                                          # [N,K]
        return {"images": images, "patch_map": patch_map, "patch_tokens": tokens,
                "encoded_tokens": encoded, "pooled": pooled, "logits": logits}

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    images = torch.randn(2, 3, 64, 64)  # N=2, 8x8 patches -> L=64 tokens
    model = TinyHyenaVision().eval()
    with torch.no_grad():
        vision_trace = model.forward_with_shapes(images)
        operator_trace = model.mixer.forward_with_shapes(vision_trace["patch_tokens"])
    print("Hyena vision path")
    for name, value in vision_trace.items():
        print(f"{name:18} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("\nHyena operator internals (no L x L attention tensor)")
    for name, value in operator_trace.items():
        print(f"{name:18} shape={tuple(value.shape)}, dtype={value.dtype}")
