"""ViR-style image classifier using multi-head retention instead of attention.

Absolute positions are added to patch embeddings before a class token is
appended, as highlighted in Unit 13. The final class token causally reads all
patches. Parallel and recurrent modes are numerically equivalent here.

Run: uv run --extra cpu python codes/unit13/vir.py
Notation: N=batch, L=patches, D=width, K=classes.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from retention import MultiScaleRetention


class VisionRetentionBlock(nn.Module):
    """Pre-norm retention and MLP residual block, `[N,L,D] -> [N,L,D]`."""

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.retention = MultiScaleRetention(embed_dim, num_heads)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
                                 nn.Linear(4 * embed_dim, embed_dim))

    def forward(self, tokens: Tensor, mode: str = "parallel") -> Tensor:
        tokens = tokens + self.retention(self.norm_1(tokens), mode=mode)
        return tokens + self.mlp(self.norm_2(tokens))


class TinyVisionRetention(nn.Module):
    """Classify fixed-size RGB images `[N,3,S,S] -> logits [N,K]`."""

    def __init__(self, image_size: int = 32, patch_size: int = 4,
                 embed_dim: int = 48, num_classes: int = 10) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.block = VisionRetentionBlock(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_with_shapes(self, images: Tensor, mode: str = "parallel"
                            ) -> dict[str, Tensor]:
        expected = (3, self.image_size, self.image_size)
        if images.ndim != 4 or images.shape[1:] != expected:
            raise ValueError(f"expected [N,{expected[0]},{expected[1]},{expected[2]}], "
                             f"got {tuple(images.shape)}")
        patch_map = self.patch_embedding(images)                             # [N,D,G,G]
        patches = patch_map.flatten(2).transpose(1, 2)                       # [N,L,D]
        positioned_patches = patches + self.position_embedding              # [N,L,D]
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)      # [N,1,D]
        # Class token is last: causal retention lets it summarize every patch.
        tokens = torch.cat((positioned_patches, class_tokens), dim=1)        # [N,L+1,D]
        encoded = self.block(tokens, mode=mode)                              # [N,L+1,D]
        class_representation = self.norm(encoded[:, -1])                    # [N,D]
        logits = self.head(class_representation)                            # [N,K]
        return {"images": images, "patch_map": patch_map, "patches": patches,
                "positioned_patches": positioned_patches, "tokens_with_cls": tokens,
                "encoded_tokens": encoded,
                "class_representation": class_representation, "logits": logits}

    def forward(self, images: Tensor, mode: str = "parallel") -> Tensor:
        return self.forward_with_shapes(images, mode)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyVisionRetention().eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        parallel_trace = model.forward_with_shapes(images, mode="parallel")
        recurrent_trace = model.forward_with_shapes(images, mode="recurrent")
    for name, value in parallel_trace.items():
        print(f"{name:22} shape={tuple(value.shape)}, dtype={value.dtype}")
    difference = (parallel_trace["logits"] - recurrent_trace["logits"]).abs().max()
    print("max parallel/recurrent logits difference =", f"{difference.item():.2e}")
