"""Vision Transformer (ViT): image -> patches -> tokens -> logits.

Run: uv run python codes/unit03/vit.py
All tensors are float32. N=batch, C=channels, H/W=spatial size,
P=patch size, E=embedding width, L=number of tokens.
"""

from __future__ import annotations
import torch
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Turn non-overlapping image patches into token vectors.

    Input: images `[N, C, H, W]`; H and W must be divisible by P.
    Output: tokens `[N, (H/P)*(W/P), E]`.

    A Conv2d with kernel=stride=P is exactly "extract each P x P patch,
    flatten it, then apply one shared linear projection", but avoids copying
    patches into a large intermediate tensor.
    """
    def __init__(self, image_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size, self.patch_size = image_size, patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"expected [N, C, {self.image_size}, {self.image_size}], got {tuple(images.shape)}")
        feature_map = self.projection(images)       # [N, E, H/P, W/P]
        return feature_map.flatten(2).transpose(1, 2)  # [N, L, E]


class VisionTransformer(nn.Module):
    """Small pre-norm ViT classifier.

    Input: `[N, 3, 32, 32]` normalized RGB images.
    Output: `[N, num_classes]` unnormalized class logits.
    """
    def __init__(self, num_classes: int = 10, embed_dim: int = 64) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(embed_dim=embed_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 reserves position zero for the class token.
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embedding.num_patches + 1, embed_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        patches = self.patch_embedding(images)               # [N, 64, E]
        cls = self.class_token.expand(images.shape[0], -1, -1)  # [N, 1, E]
        tokens = torch.cat((cls, patches), dim=1)             # [N, 65, E]
        positioned = tokens + self.position_embedding         # shape unchanged
        encoded = self.encoder(positioned)                    # global token mixing
        cls_output = self.norm(encoded[:, 0])                 # [N, E]
        logits = self.head(cls_output)                        # [N, classes]
        return {"images": images, "patch_tokens": patches, "tokens_with_cls": tokens,
                "positioned_tokens": positioned, "encoded_tokens": encoded,
                "class_representation": cls_output, "logits": logits}

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = VisionTransformer().eval()
    with torch.no_grad():
        trace = model.forward_with_shapes(torch.randn(2, 3, 32, 32))
    print("ViT: 32x32 image / 4x4 patch = 8x8 = 64 patch tokens")
    for name, value in trace.items():
        print(f"{name:24} {tuple(value.shape)}")
