"""Layout-aware Document VQA, inspired by LayoutLM.

Run with:
    uv run python codes/unit04/document_vqa.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LayoutAwareDocumentVQA(nn.Module):
    """Predict an answer span from OCR words, positions, and a question.

    Inputs:
        word_ids: OCR token IDs `[batch, document_length]`.
        boxes: normalized integer boxes `[batch, document_length, 4]`, ordered
            as `(x_min, y_min, x_max, y_max)`, each coordinate in `[0, 1000]`.
        question_ids: token IDs `[batch, question_length]`.
    Outputs:
        start/end logits `[batch, document_length]`.

    This demonstrates the LayoutLM path. Donut and Nougat instead skip OCR and
    pass image patches to an autoregressive text decoder (as in captioning.py).
    """

    def __init__(self, vocabulary_size: int = 200, hidden_dim: int = 32) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocabulary_size, hidden_dim)
        self.coordinate_embedding = nn.Embedding(1001, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.document_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.question_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.span_head = nn.Linear(hidden_dim, 2)

    def forward_with_shapes(
        self, word_ids: Tensor, boxes: Tensor, question_ids: Tensor
    ) -> dict[str, Tensor]:
        if boxes.shape != (*word_ids.shape, 4):
            raise ValueError(f"boxes must be {(*word_ids.shape, 4)}, got {tuple(boxes.shape)}")
        if boxes.min() < 0 or boxes.max() > 1000:
            raise ValueError("box coordinates must be normalized to [0, 1000]")
        words = self.word_embedding(word_ids)  # [N, Ld] -> [N, Ld, D]
        layout = self.coordinate_embedding(boxes).sum(dim=2)  # [N, Ld, 4, D] -> [N, Ld, D]
        document = self.document_encoder(words + layout)  # [N, Ld, D]
        question_words = self.word_embedding(question_ids)  # [N, Lq, D]
        _, question_state = self.question_encoder(question_words)  # state [1, N, D]
        question = question_state.transpose(0, 1)  # [N, 1, D]
        attended, attention = self.cross_attention(
            query=document, key=question, value=question
        )  # attended [N, Ld, D], attention [N, Ld, 1]
        span_logits = self.span_head(document + attended)  # [N, Ld, 2]
        return {
            "word_ids": word_ids,
            "boxes": boxes,
            "word_embeddings": words,
            "layout_embeddings": layout,
            "document_tokens": document,
            "question_vector": question,
            "question_attention": attention,
            "start_logits": span_logits[..., 0],
            "end_logits": span_logits[..., 1],
        }

    def forward(self, word_ids: Tensor, boxes: Tensor, question_ids: Tensor) -> tuple[Tensor, Tensor]:
        steps = self.forward_with_shapes(word_ids, boxes, question_ids)
        return steps["start_logits"], steps["end_logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = LayoutAwareDocumentVQA().eval()
    words = torch.randint(0, 200, (2, 12), dtype=torch.long)
    boxes = torch.randint(0, 1001, (2, 12, 4), dtype=torch.long)
    questions = torch.randint(0, 200, (2, 5), dtype=torch.long)
    with torch.no_grad():
        steps = model.forward_with_shapes(words, boxes, questions)
    for name, tensor in steps.items():
        print(f"{name:20} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    starts = steps["start_logits"].argmax(dim=-1)
    ends = steps["end_logits"].argmax(dim=-1)
    print("predicted OCR answer spans:", list(zip(starts.tolist(), ends.tolist())))
