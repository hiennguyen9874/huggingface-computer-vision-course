"""Bidirectional image-text retrieval in a shared embedding space.

Run with:
    uv run python codes/unit04/image_text_retrieval.py
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


class MultimodalIndex:
    """An exact, in-memory vector index for teaching retrieval mechanics.

    Stored image/text embeddings have shape `[items, embedding_dim]`. Queries
    have shape `[queries, embedding_dim]`; returned indices and scores are
    `[queries, k]`. Production systems replace the exact matrix multiply with a
    vector database or approximate nearest-neighbor index.
    """

    def __init__(self, image_embeddings: Tensor, text_embeddings: Tensor) -> None:
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [items, embedding_dim]")
        if image_embeddings.shape[1] != text_embeddings.shape[1]:
            raise ValueError("image and text embedding dimensions must match")
        self.image_embeddings = F.normalize(image_embeddings, dim=-1)
        self.text_embeddings = F.normalize(text_embeddings, dim=-1)

    @staticmethod
    def _top_k(query: Tensor, candidates: Tensor, k: int) -> tuple[Tensor, Tensor]:
        similarities = F.normalize(query, dim=-1) @ candidates.T  # [Nq, Nc]
        scores, indices = similarities.topk(min(k, candidates.shape[0]), dim=-1)
        return indices, scores  # both [Nq, k]

    def text_to_image(self, text_queries: Tensor, k: int = 2) -> tuple[Tensor, Tensor]:
        """Retrieve image indices for text queries `[queries, embedding_dim]`."""
        return self._top_k(text_queries, self.image_embeddings, k)

    def image_to_text(self, image_queries: Tensor, k: int = 2) -> tuple[Tensor, Tensor]:
        """Retrieve caption indices for image queries `[queries, embedding_dim]`."""
        return self._top_k(image_queries, self.text_embeddings, k)


if __name__ == "__main__":
    torch.manual_seed(0)
    database_images = F.normalize(torch.randn(5, 16), dim=-1)
    # Captions are near their paired images, simulating CLIP-aligned embeddings.
    database_captions = database_images + 0.05 * torch.randn(5, 16)
    index = MultimodalIndex(database_images, database_captions)

    text_queries = database_captions[[3, 1]]  # [2 queries, 16 features]
    image_indices, image_scores = index.text_to_image(text_queries, k=3)
    print(f"stored images      {tuple(index.image_embeddings.shape)}")
    print(f"text queries       {tuple(text_queries.shape)}")
    print(f"image indices      {tuple(image_indices.shape)}\n{image_indices}")
    print(f"image scores       {tuple(image_scores.shape)}\n{image_scores.round(decimals=3)}")

    image_queries = database_images[[4]]
    text_indices, text_scores = index.image_to_text(image_queries, k=3)
    print(f"image queries      {tuple(image_queries.shape)}")
    print(f"caption indices    {tuple(text_indices.shape)}\n{text_indices}")
    print(f"caption scores     {tuple(text_scores.shape)}\n{text_scores.round(decimals=3)}")
