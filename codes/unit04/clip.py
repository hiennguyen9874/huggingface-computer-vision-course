"""A compact CLIP-style dual encoder and zero-shot classifier.

This is an architectural lesson, not a pretrained OpenAI CLIP checkpoint.
Run with:
    uv run python codes/unit04/clip.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TinyImageEncoder(nn.Module):
    """Encode RGB images `[N, 3, H, W]` as vectors `[N, output_dim]`."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=4),  # [N, 3, 32, 32] -> [N, 32, 8, 8]
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),  # -> [N, 64, 4, 4]
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # -> [N, 64, 1, 1]
        )
        self.projection = nn.Linear(64, output_dim)

    def forward(self, images: Tensor) -> Tensor:
        return self.projection(self.features(images).flatten(1))  # [N, output_dim]


class TinyTextEncoder(nn.Module):
    """Encode integer token IDs `[N, length]` as vectors `[N, output_dim]`."""

    def __init__(self, vocabulary_size: int, output_dim: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, 32)
        layer = nn.TransformerEncoderLayer(32, nhead=4, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.projection = nn.Linear(32, output_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        tokens = self.token_embedding(token_ids)  # [N, L] -> [N, L, 32]
        contextualized = self.encoder(tokens)  # [N, L, 32]
        return self.projection(contextualized.mean(dim=1))  # [N, output_dim]


class EducationalCLIP(nn.Module):
    """Map images and text into one L2-normalized embedding space.

    Inputs:
        images: float RGB tensor `[image_batch, 3, 32, 32]`.
        token_ids: integer tensor `[text_batch, text_length]`.
    Outputs:
        image/text embeddings `[image_batch, D]` and `[text_batch, D]`;
        all-pairs similarity logits `[image_batch, text_batch]`.
    """

    def __init__(self, vocabulary_size: int = 100, embedding_dim: int = 32) -> None:
        super().__init__()
        self.image_encoder = TinyImageEncoder(embedding_dim)
        self.text_encoder = TinyTextEncoder(vocabulary_size, embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0).log())

    def forward_with_shapes(self, images: Tensor, token_ids: Tensor) -> dict[str, Tensor]:
        image_raw = self.image_encoder(images)  # [Ni, 3, H, W] -> [Ni, D]
        text_raw = self.text_encoder(token_ids)  # [Nt, L] -> [Nt, D]
        image_embeddings = F.normalize(image_raw, dim=-1)  # unit vectors [Ni, D]
        text_embeddings = F.normalize(text_raw, dim=-1)  # unit vectors [Nt, D]
        logits = self.logit_scale.exp() * image_embeddings @ text_embeddings.T  # [Ni, Nt]
        return {
            "images": images,
            "token_ids": token_ids,
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "similarity_logits": logits,
            "class_probabilities": logits.softmax(dim=-1),
        }

    def forward(self, images: Tensor, token_ids: Tensor) -> Tensor:
        return self.forward_with_shapes(images, token_ids)["similarity_logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = EducationalCLIP().eval()
    images = torch.randn(2, 3, 32, 32)
    # Three tokenized natural-language prompts, e.g. cat, dog, and car templates.
    prompt_token_ids = torch.randint(0, 100, (3, 6), dtype=torch.long)
    with torch.no_grad():
        steps = model.forward_with_shapes(images, prompt_token_ids)
    for name, tensor in steps.items():
        print(f"{name:20} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print("zero-shot probabilities per image:\n", steps["class_probabilities"].round(decimals=3))
