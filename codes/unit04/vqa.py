"""A compact Visual Question Answering model with cross-attention.

Run with:
    uv run python codes/unit04/vqa.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EducationalVQA(nn.Module):
    """Treat VQA as classification over a fixed answer vocabulary.

    Inputs:
        images: float RGB tensor `[batch, 3, 32, 32]`.
        questions: integer token IDs `[batch, question_length]`.
    Output:
        answer logits `[batch, number_of_answers]`.

    The image becomes patch tokens. Question tokens query those patches through
    cross-attention, which grounds the answer representation in visual content.
    """

    def __init__(
        self, vocabulary_size: int = 100, number_of_answers: int = 8, hidden_dim: int = 32
    ) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=8, stride=8)
        self.word_embedding = nn.Embedding(vocabulary_size, hidden_dim)
        self.question_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.answer_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, number_of_answers)
        )

    def forward_with_shapes(self, images: Tensor, questions: Tensor) -> dict[str, Tensor]:
        patch_grid = self.patch_embedding(images)  # [N, 3, 32, 32] -> [N, D, 4, 4]
        image_tokens = patch_grid.flatten(2).transpose(1, 2)  # -> [N, 16, D]
        question_tokens = self.word_embedding(questions)  # [N, L] -> [N, L, D]
        question_context, _ = self.question_encoder(question_tokens)  # [N, L, D]
        grounded, attention_weights = self.cross_attention(
            query=question_context, key=image_tokens, value=image_tokens
        )  # grounded [N, L, D], weights [N, L, 16]
        fused = (question_context + grounded).mean(dim=1)  # [N, D]
        logits = self.answer_head(fused)  # [N, number_of_answers]
        return {
            "images": images,
            "questions": questions,
            "patch_grid": patch_grid,
            "image_tokens": image_tokens,
            "question_tokens": question_tokens,
            "grounded_question": grounded,
            "attention_weights": attention_weights,
            "fused": fused,
            "answer_logits": logits,
        }

    def forward(self, images: Tensor, questions: Tensor) -> Tensor:
        return self.forward_with_shapes(images, questions)["answer_logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = EducationalVQA().eval()
    images = torch.randn(2, 3, 32, 32)
    questions = torch.randint(0, 100, (2, 7), dtype=torch.long)
    with torch.no_grad():
        steps = model.forward_with_shapes(images, questions)
    for name, tensor in steps.items():
        print(f"{name:20} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print("predicted answer IDs:", steps["answer_logits"].argmax(dim=-1).tolist())
