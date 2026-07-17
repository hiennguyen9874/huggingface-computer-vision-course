"""OneFormer concepts: task conditioning and query-text contrastive learning.

Run: uv run python codes/unit03/oneformer.py
A full OneFormer is large; this compact model exposes its defining contracts.
"""

from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F

TASKS = {"semantic": 0, "instance": 1, "panoptic": 2}


class TinyOneFormer(nn.Module):
    """One set of weights conditioned on semantic/instance/panoptic task input.

    Inputs: images `[N,3,H,W]`; task_ids int64 `[N]` in {0,1,2}.
    Outputs: class logits `[N,Q,K+1]`, mask logits `[N,Q,H/4,W/4]`, and
    normalized visual query embeddings `[N,Q,D]` for contrastive training.
    """
    def __init__(self, num_classes: int = 3, num_queries: int = 6, dim: int = 64) -> None:
        super().__init__(); self.num_queries = num_queries
        self.pixels = nn.Sequential(nn.Conv2d(3, dim, 4, 4), nn.GELU(), nn.Conv2d(dim, dim, 3, padding=1))
        self.task_embedding, self.queries = nn.Embedding(3, dim), nn.Embedding(num_queries, dim)
        layer = nn.TransformerDecoderLayer(dim, 4, 4*dim, dropout=0., batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.class_head, self.mask_head = nn.Linear(dim, num_classes+1), nn.Linear(dim, dim)

    def forward_with_shapes(self, images: Tensor, task_ids: Tensor) -> dict[str, Tensor]:
        if task_ids.shape != (images.shape[0],): raise ValueError("task_ids must have shape [N]")
        pixels = self.pixels(images); memory = pixels.flatten(2).transpose(1, 2)
        base_queries = self.queries.weight.unsqueeze(0).expand(images.shape[0], -1, -1)
        task_tokens = self.task_embedding(task_ids).unsqueeze(1)    # [N,1,D]
        guided_queries = base_queries + task_tokens                # task changes every query
        decoded = self.decoder(guided_queries, memory)
        mask_embeddings = self.mask_head(decoded)
        mask_logits = torch.einsum("nqd,ndhw->nqhw", mask_embeddings, pixels)
        return {"images": images, "task_tokens": task_tokens, "task_guided_queries": guided_queries,
                "pixel_embeddings": pixels, "decoded_queries": decoded,
                "class_logits": self.class_head(decoded), "mask_logits": mask_logits,
                "normalized_queries": F.normalize(decoded, dim=-1)}

    def forward(self, images: Tensor, task_ids: Tensor) -> tuple[Tensor, Tensor]:
        trace = self.forward_with_shapes(images, task_ids); return trace["class_logits"], trace["mask_logits"]


def query_text_contrastive_loss(visual_queries: Tensor, text_embeddings: Tensor, temperature: float = 0.1) -> Tensor:
    """Align pooled visual queries `[N,Q,D]` with matching text `[N,D]`.

    Similarity logits are `[N,N]`; diagonal entries are positive image-text pairs.
    """
    visual = F.normalize(visual_queries.mean(dim=1), dim=-1); text = F.normalize(text_embeddings, dim=-1)
    similarities = visual @ text.transpose(0, 1) / temperature
    labels = torch.arange(visual.shape[0], device=visual.device)
    return (F.cross_entropy(similarities, labels) + F.cross_entropy(similarities.T, labels)) / 2


if __name__ == "__main__":
    torch.manual_seed(0); images = torch.randn(3, 3, 32, 32); task_ids = torch.tensor(list(TASKS.values()))
    model = TinyOneFormer().eval()
    with torch.no_grad(): trace = model.forward_with_shapes(images, task_ids)
    print("tasks:", list(TASKS));
    for name, tensor in trace.items(): print(f"{name:22} {tuple(tensor.shape)}")
    text = model.task_embedding(task_ids)
    print(f"query-text contrastive loss: {query_text_contrastive_loss(trace['normalized_queries'], text).item():.4f}")
