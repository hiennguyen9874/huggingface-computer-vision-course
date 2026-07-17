"""BLIP's shared vision encoder and mixture of encoder-decoder heads.

Run with:
    uv run python codes/unit04/blip.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EducationalBLIP(nn.Module):
    """Expose BLIP's three objectives in one compact model.

    Inputs:
        images: float RGB tensor `[batch, 3, 32, 32]`.
        text_ids: integer token IDs `[batch, text_length]`.
    Outputs:
        normalized image/text embeddings `[batch, D]` for contrastive learning;
        matching logits `[batch, 2]` for image-text matching;
        generation logits `[batch, text_length, vocabulary_size]`.

    Real BLIP shares selected MED parameters. This lesson keeps the paths named
    and visible while using one small image encoder and one word embedding.
    """

    def __init__(self, vocabulary_size: int = 100, hidden_dim: int = 32) -> None:
        super().__init__()
        self.vision_encoder = nn.Conv2d(3, hidden_dim, kernel_size=8, stride=8)
        self.word_embedding = nn.Embedding(vocabulary_size, hidden_dim)
        text_layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 2, batch_first=True)
        self.unimodal_text_encoder = nn.TransformerEncoder(text_layer, num_layers=1)
        self.grounded_encoder_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.matching_head = nn.Linear(hidden_dim, 2)
        self.grounded_decoder = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.generation_head = nn.Linear(hidden_dim, vocabulary_size)

    def forward_with_shapes(self, images: Tensor, text_ids: Tensor) -> dict[str, Tensor]:
        patch_grid = self.vision_encoder(images)  # [N, 3, 32, 32] -> [N, D, 4, 4]
        visual_tokens = patch_grid.flatten(2).transpose(1, 2)  # [N, 16, D]
        word_tokens = self.word_embedding(text_ids)  # [N, L] -> [N, L, D]
        text_tokens = self.unimodal_text_encoder(word_tokens)  # [N, L, D]

        image_embedding = F.normalize(visual_tokens.mean(dim=1), dim=-1)  # [N, D]
        text_embedding = F.normalize(text_tokens.mean(dim=1), dim=-1)  # [N, D]
        contrastive_logits = image_embedding @ text_embedding.T  # [N, N]

        grounded_tokens, _ = self.grounded_encoder_attention(
            query=text_tokens, key=visual_tokens, value=visual_tokens
        )  # [N, L, D]
        matching_logits = self.matching_head(grounded_tokens.mean(dim=1))  # [N, 2]

        visual_context = visual_tokens.mean(dim=1, keepdim=True)  # [N, 1, D]
        decoder_input = torch.cat(
            (word_tokens, visual_context.expand(-1, word_tokens.shape[1], -1)), dim=-1
        )  # [N, L, 2D]
        decoder_states, _ = self.grounded_decoder(decoder_input)  # [N, L, D]
        generation_logits = self.generation_head(decoder_states)  # [N, L, V]
        return {
            "patch_grid": patch_grid,
            "visual_tokens": visual_tokens,
            "text_tokens": text_tokens,
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "contrastive_logits": contrastive_logits,
            "grounded_tokens": grounded_tokens,
            "matching_logits": matching_logits,
            "decoder_states": decoder_states,
            "generation_logits": generation_logits,
        }

    def forward(self, images: Tensor, text_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        steps = self.forward_with_shapes(images, text_ids)
        return steps["contrastive_logits"], steps["matching_logits"], steps["generation_logits"]


def capfilt_keep_mask(
    original_match_scores: Tensor, synthetic_match_scores: Tensor, threshold: float
) -> Tensor:
    """Model CapFilt's filtering decision.

    Inputs are caption-image match scores `[items]` for noisy web captions and
    generated captions. Output `[items]` is true when either caption is useful.
    """
    if original_match_scores.shape != synthetic_match_scores.shape:
        raise ValueError("original and synthetic score shapes must match")
    return torch.maximum(original_match_scores, synthetic_match_scores) >= threshold


if __name__ == "__main__":
    torch.manual_seed(0)
    model = EducationalBLIP().eval()
    images = torch.randn(3, 3, 32, 32)
    text_ids = torch.randint(0, 100, (3, 7), dtype=torch.long)
    with torch.no_grad():
        steps = model.forward_with_shapes(images, text_ids)
    for name, tensor in steps.items():
        print(f"{name:20} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

    original = torch.tensor([0.9, 0.2, 0.3, 0.1])
    synthetic = torch.tensor([0.8, 0.7, 0.4, 0.2])
    keep = capfilt_keep_mask(original, synthetic, threshold=0.6)
    print(f"CapFilt scores       shape={tuple(original.shape)}")
    print("CapFilt keep mask:  ", keep.tolist())
