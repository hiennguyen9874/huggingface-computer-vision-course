"""DETR: CNN features + encoder/decoder + fixed object-query predictions.

Run: uv run python codes/unit03/detr.py
This is a compact architecture lesson, not a pretrained detector.
"""

from __future__ import annotations
import torch
from torch import Tensor, nn


class TinyDETR(nn.Module):
    """Direct set prediction without anchors or non-maximum suppression.

    Input: images `[N,3,H,W]` (demo uses 64x64).
    Outputs:
      class_logits `[N,Q,K+1]`: K object classes plus final no-object class.
      boxes `[N,Q,4]`: normalized `(center_x, center_y, width, height)` in [0,1].
    Q is fixed; unmatched predictions are trained as no-object.
    """
    def __init__(self, num_classes: int = 3, num_queries: int = 10, hidden_dim: int = 64) -> None:
        super().__init__(); self.num_queries = num_queries
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, hidden_dim, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1), nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 4, 4*hidden_dim, dropout=0., batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, 4, 4*hidden_dim, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.decoder = nn.TransformerDecoder(decoder_layer, 2)
        self.row_position, self.column_position = nn.Embedding(32, hidden_dim//2), nn.Embedding(32, hidden_dim//2)
        self.object_queries = nn.Embedding(num_queries, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.box_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4))

    def positional_encoding(self, h: int, w: int, batch: int) -> Tensor:
        rows = self.row_position(torch.arange(h, device=self.row_position.weight.device))
        columns = self.column_position(torch.arange(w, device=self.column_position.weight.device))
        grid = torch.cat((columns.unsqueeze(0).expand(h, -1, -1),
                          rows.unsqueeze(1).expand(-1, w, -1)), dim=-1)
        return grid.reshape(1, h*w, -1).expand(batch, -1, -1)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        feature_map = self.backbone(images)                       # [N,D,H',W']
        n, _, h, w = feature_map.shape
        sequence = feature_map.flatten(2).transpose(1, 2)        # [N,H'*W',D]
        positioned = sequence + self.positional_encoding(h, w, n)
        memory = self.encoder(positioned)                         # encoder image memory
        queries = self.object_queries.weight.unsqueeze(0).expand(n, -1, -1)  # [N,Q,D]
        decoded = self.decoder(queries, memory)                   # each query reads image
        class_logits = self.class_head(decoded)                   # [N,Q,K+1]
        boxes = self.box_head(decoded).sigmoid()                  # [N,Q,4], normalized
        return {"images": images, "cnn_feature_map": feature_map, "feature_sequence": sequence,
                "encoder_memory": memory, "object_queries": queries, "decoder_output": decoded,
                "class_logits": class_logits, "boxes_cxcywh": boxes}

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        trace = self.forward_with_shapes(images)
        return trace["class_logits"], trace["boxes_cxcywh"]


def cxcywh_to_xyxy(boxes: Tensor, image_height: int, image_width: int) -> Tensor:
    """Normalized `[...,(cx,cy,w,h)]` -> pixel `[...,(x1,y1,x2,y2)]`."""
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(((cx-width/2)*image_width, (cy-height/2)*image_height,
                        (cx+width/2)*image_width, (cy+height/2)*image_height), dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    with torch.no_grad(): trace = TinyDETR().eval().forward_with_shapes(torch.randn(2, 3, 64, 64))
    for name, tensor in trace.items(): print(f"{name:20} {tuple(tensor.shape)}")
    print("first query pixel xyxy:", cxcywh_to_xyxy(trace["boxes_cxcywh"][0, 0], 64, 64).tolist())
