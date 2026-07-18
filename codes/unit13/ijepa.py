"""I-JEPA: predict masked target embeddings from visible context embeddings.

This compact training example shows the three defining modules: context encoder,
EMA target encoder, and predictor. It predicts representations, never pixels.
It is educational code, not Meta's paper-scale training recipe.

Run: uv run --extra cpu python codes/unit13/ijepa.py
Notation: N=batch, L=patches, C=context patches, T=target patches, E=embedding.
"""
from __future__ import annotations

import copy
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PatchEncoder(nn.Module):
    """Encode selected image patches with a small ViT.

    `embed_patches` maps `[N,3,H,W] -> [N,L,E]`. `encode_selected` receives
    patch embeddings `[N,L,E]` and indices `[S]`, then returns `[N,S,E]`.
    """

    def __init__(self, image_size: int = 32, patch_size: int = 4,
                 embed_dim: int = 48, depth: int = 2) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_projection = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        layer = nn.TransformerEncoderLayer(
            embed_dim, nhead=4, dim_feedforward=4 * embed_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def embed_patches(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(
                f"expected [N,3,{self.image_size},{self.image_size}], got {tuple(images.shape)}"
            )
        patch_map = self.patch_projection(images)                           # [N,E,G,G]
        return patch_map.flatten(2).transpose(1, 2)                         # [N,L,E]

    def encode_selected(self, patches: Tensor, indices: Tensor) -> Tensor:
        if patches.ndim != 3 or patches.shape[1:] != (self.num_patches, self.embed_dim):
            raise ValueError(f"expected patches [N,{self.num_patches},{self.embed_dim}]")
        selected = patches.index_select(1, indices)                         # [N,S,E]
        positions = self.position_embedding.index_select(1, indices)       # [1,S,E]
        return self.norm(self.blocks(selected + positions))                 # [N,S,E]

    def forward(self, images: Tensor, indices: Tensor) -> Tensor:
        return self.encode_selected(self.embed_patches(images), indices)


class JEPAPredictor(nn.Module):
    """Predict embeddings at T target positions from C context embeddings.

    Inputs: context `[N,C,E]`, context indices `[C]`, target indices `[T]`.
    Output: predicted target representations `[N,T,E]`.
    """

    def __init__(self, num_patches: int, embed_dim: int) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        layer = nn.TransformerEncoderLayer(
            embed_dim, nhead=4, dim_feedforward=4 * embed_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context: Tensor, context_indices: Tensor,
                target_indices: Tensor) -> Tensor:
        batch_size, context_count, width = context.shape
        if width != self.embed_dim or context_count != context_indices.numel():
            raise ValueError("context shape must agree with context_indices and embed_dim")
        target_count = target_indices.numel()
        context_pos = self.position_embedding.index_select(1, context_indices) # [1,C,E]
        target_pos = self.position_embedding.index_select(1, target_indices)   # [1,T,E]
        target_queries = self.mask_token.expand(batch_size, target_count, -1) + target_pos
        predictor_input = torch.cat((context + context_pos, target_queries), dim=1) # [N,C+T,E]
        predicted_all = self.norm(self.blocks(predictor_input))                    # [N,C+T,E]
        return predicted_all[:, context_count:]                                    # [N,T,E]


def rectangular_patch_indices(grid_size: int, top: int, left: int,
                              height: int, width: int) -> Tensor:
    """Return flattened row-major indices `[height*width]` for one 2-D block."""
    if min(top, left, height, width) < 0 or height == 0 or width == 0:
        raise ValueError("block coordinates must be non-negative and size must be positive")
    if top + height > grid_size or left + width > grid_size:
        raise ValueError("target block extends outside the patch grid")
    return torch.tensor([(top + row) * grid_size + left + column
                         for row in range(height) for column in range(width)])


class TinyIJEPA(nn.Module):
    """Self-supervised I-JEPA training system returning a scalar embedding loss."""

    def __init__(self, image_size: int = 32, patch_size: int = 4,
                 embed_dim: int = 48) -> None:
        super().__init__()
        self.context_encoder = PatchEncoder(image_size, patch_size, embed_dim)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = JEPAPredictor(self.context_encoder.num_patches, embed_dim)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996) -> None:
        """EMA update: target <- momentum*target + (1-momentum)*context."""
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must be in [0,1]")
        for target, context in zip(self.target_encoder.parameters(),
                                   self.context_encoder.parameters()):
            target.mul_(momentum).add_(context, alpha=1 - momentum)

    def forward_with_shapes(self, images: Tensor, context_indices: Tensor,
                            target_indices: Tensor) -> dict[str, Tensor]:
        overlap = torch.isin(context_indices, target_indices)
        if overlap.any():
            raise ValueError("context_indices and target_indices must not overlap")
        context_patches = self.context_encoder.embed_patches(images)        # [N,L,E]
        context = self.context_encoder.encode_selected(context_patches, context_indices)
        with torch.no_grad():
            target_patches = self.target_encoder.embed_patches(images)      # [N,L,E]
            target = self.target_encoder.encode_selected(target_patches, target_indices)
        predicted_target = self.predictor(context, context_indices, target_indices)
        loss = F.smooth_l1_loss(predicted_target, target)                    # scalar
        return {"images": images, "all_patch_embeddings": context_patches,
                "context_embeddings": context, "target_embeddings": target,
                "predicted_target": predicted_target, "embedding_loss": loss}

    def forward(self, images: Tensor, context_indices: Tensor,
                target_indices: Tensor) -> Tensor:
        return self.forward_with_shapes(images, context_indices, target_indices)["embedding_loss"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyIJEPA()
    # On an 8x8 patch grid, use one large 4x4 target block. All other 48 patches
    # are context. Real I-JEPA samples several varied target blocks per image.
    target_indices = rectangular_patch_indices(8, top=2, left=2, height=4, width=4)
    all_indices = torch.arange(64)
    context_indices = all_indices[~torch.isin(all_indices, target_indices)]
    images = torch.randn(2, 3, 32, 32)

    trace = model.forward_with_shapes(images, context_indices, target_indices)
    trace["embedding_loss"].backward()  # gradients: context encoder + predictor only
    model.update_target_encoder()

    print(f"context indices shape={tuple(context_indices.shape)}; "
          f"target block indices shape={tuple(target_indices.shape)}")
    for name, value in trace.items():
        print(f"{name:22} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("target encoder has gradients:",
          any(parameter.grad is not None for parameter in model.target_encoder.parameters()))
