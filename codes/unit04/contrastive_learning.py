"""Pairwise contrastive loss and CLIP's symmetric batch objective.

Run with:
    uv run python codes/unit04/contrastive_learning.py
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def pairwise_contrastive_loss(
    first: Tensor, second: Tensor, same_class: Tensor, margin: float = 1.0
) -> Tensor:
    """Compute the classic margin-based contrastive loss.

    Inputs:
        first, second: float embeddings `[batch, embedding_dim]`.
        same_class: bool tensor `[batch]`; true means a positive pair.
        margin: minimum Euclidean distance requested for negative pairs.
    Output:
        Scalar mean loss `[]`.
    """
    if first.shape != second.shape:
        raise ValueError(f"embedding shapes must match, got {first.shape} and {second.shape}")
    distances = torch.linalg.vector_norm(first - second, dim=-1)  # [N]
    positive_loss = distances.square()  # pull matching samples together
    negative_loss = F.relu(margin - distances).square()  # push negatives to margin
    return torch.where(same_class.bool(), positive_loss, negative_loss).mean()


def clip_similarity_and_loss(
    image_embeddings: Tensor, text_embeddings: Tensor, temperature: float = 0.07
) -> tuple[Tensor, Tensor]:
    """Return all-pairs CLIP logits and symmetric cross-entropy loss.

    Row `i` and column `i` must describe a matching image-text pair.

    Inputs: image/text float embeddings `[batch, embedding_dim]`.
    Outputs: similarity logits `[batch, batch]`, scalar loss `[]`.
    """
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError(
            "paired image/text embeddings need equal shapes; "
            f"got {image_embeddings.shape} and {text_embeddings.shape}"
        )
    image = F.normalize(image_embeddings, dim=-1)  # [N, D]
    text = F.normalize(text_embeddings, dim=-1)  # [N, D]
    logits = image @ text.transpose(0, 1) / temperature  # [N, N]
    targets = torch.arange(logits.shape[0], device=logits.device)  # [N]
    image_to_text = F.cross_entropy(logits, targets)  # classify caption per image
    text_to_image = F.cross_entropy(logits.transpose(0, 1), targets)
    return logits, (image_to_text + text_to_image) / 2


if __name__ == "__main__":
    torch.manual_seed(0)
    first = torch.randn(4, 8)
    second = first + 0.1 * torch.randn(4, 8)
    labels = torch.tensor([True, True, False, False])
    pair_loss = pairwise_contrastive_loss(first, second, labels)
    print(f"pair embeddings       {tuple(first.shape)}")
    print(f"pair labels           {tuple(labels.shape)}")
    print(f"pairwise loss         shape={tuple(pair_loss.shape)}, value={pair_loss.item():.4f}")

    image_embeddings = torch.randn(3, 16)
    text_embeddings = image_embeddings + 0.05 * torch.randn(3, 16)
    logits, clip_loss = clip_similarity_and_loss(image_embeddings, text_embeddings)
    print(f"CLIP image embeddings {tuple(image_embeddings.shape)}")
    print(f"CLIP text embeddings  {tuple(text_embeddings.shape)}")
    print(f"similarity matrix     {tuple(logits.shape)}\n{logits.round(decimals=2)}")
    print(f"symmetric loss        shape={tuple(clip_loss.shape)}, value={clip_loss.item():.4f}")
