"""MaskFormer: pixel embeddings + segment queries -> classes and binary masks.

Run: uv run python codes/unit03/maskformer.py
"""

from __future__ import annotations
import torch
from torch import Tensor, nn


class TinyMaskFormer(nn.Module):
    """Query-based segmentation shared by semantic/instance-style prediction.

    Input: images `[N,3,H,W]` (demo: 32x32).
    Outputs:
      class_logits `[N,Q,K+1]` (last class means no segment),
      mask_logits `[N,Q,H/4,W/4]` (one binary mask per query).
    A dot product combines each query mask embedding `[N,Q,D]` with every
    pixel embedding `[N,D,h,w]`; sigmoid is applied only for probabilities.
    """
    def __init__(self, num_classes: int = 3, num_queries: int = 6, dim: int = 64) -> None:
        super().__init__()
        self.pixel_module = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, dim, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        decoder_layer = nn.TransformerDecoderLayer(dim, 4, 4*dim, dropout=0., batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 2)
        self.segment_queries = nn.Embedding(num_queries, dim)
        self.class_head, self.mask_head = nn.Linear(dim, num_classes + 1), nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        pixels = self.pixel_module(images)                         # [N,D,h,w]
        memory = pixels.flatten(2).transpose(1, 2)                 # [N,h*w,D]
        queries = self.segment_queries.weight.unsqueeze(0).expand(images.shape[0], -1, -1)
        segments = self.decoder(queries, memory)                   # [N,Q,D]
        class_logits = self.class_head(segments)                   # [N,Q,K+1]
        mask_embeddings = self.mask_head(segments)                 # [N,Q,D]
        mask_logits = torch.einsum("nqd,ndhw->nqhw", mask_embeddings, pixels)
        return {"images": images, "pixel_embeddings": pixels, "pixel_sequence": memory,
                "segment_queries": queries, "segment_embeddings": segments,
                "class_logits": class_logits, "mask_embeddings": mask_embeddings,
                "mask_logits": mask_logits, "mask_probabilities": mask_logits.sigmoid()}

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        trace = self.forward_with_shapes(images); return trace["class_logits"], trace["mask_logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    with torch.no_grad(): trace = TinyMaskFormer().eval().forward_with_shapes(torch.randn(2, 3, 32, 32))
    for name, tensor in trace.items(): print(f"{name:20} {tuple(tensor.shape)}")
    print("Masks can be bilinearly upsampled from 8x8 to the original 32x32 resolution.")
