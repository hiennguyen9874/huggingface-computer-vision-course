"""Object-detection preprocessing, variable-size collation, matching, and loss.

Run: uv run python codes/unit03/detection_training.py
No image library is needed: synthetic tensors make all contracts observable.
"""

from __future__ import annotations
from itertools import permutations
import torch
from torch import Tensor
from torch.nn import functional as F


def coco_xywh_to_normalized_cxcywh(boxes: Tensor, height: int, width: int) -> Tensor:
    """COCO pixel boxes `[M,(x,y,w,h)]` -> normalized DETR boxes `[M,(cx,cy,w,h)]`."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack(((x+w/2)/width, (y+h/2)/height, w/width, h/height), dim=-1)


def horizontal_flip(image: Tensor, boxes_xywh: Tensor) -> tuple[Tensor, Tensor]:
    """Flip both image `[C,H,W]` and COCO boxes `[M,4]`; never augment only one."""
    flipped_boxes = boxes_xywh.clone()
    flipped_boxes[:, 0] = image.shape[-1] - boxes_xywh[:, 0] - boxes_xywh[:, 2]
    return image.flip(-1), flipped_boxes


def collate_detection_batch(samples: list[dict[str, Tensor]]) -> dict[str, object]:
    """Pad variable H/W images and retain variable-length labels.

    Returns pixel_values `[N,C,maxH,maxW]`, pixel_mask `[N,maxH,maxW]` (True
    means real pixel), and labels as a list because each image has M_i objects.
    """
    max_h = max(int(s["image"].shape[-2]) for s in samples)
    max_w = max(int(s["image"].shape[-1]) for s in samples)
    n, channels = len(samples), int(samples[0]["image"].shape[0])
    values = torch.zeros(n, channels, max_h, max_w); mask = torch.zeros(n, max_h, max_w, dtype=torch.bool)
    labels: list[dict[str, Tensor]] = []
    for i, sample in enumerate(samples):
        image = sample["image"]; h, w = image.shape[-2:]
        values[i, :, :h, :w], mask[i, :h, :w] = image, True
        labels.append({"class_labels": sample["class_labels"],
                       "boxes": coco_xywh_to_normalized_cxcywh(sample["boxes_xywh"], h, w)})
    return {"pixel_values": values, "pixel_mask": mask, "labels": labels}


def tiny_bipartite_match(class_logits: Tensor, predicted_boxes: Tensor,
                          target_classes: Tensor, target_boxes: Tensor) -> tuple[Tensor, Tensor]:
    """Exact minimum-cost matching for tiny demonstrations (M<=8), not production.

    Inputs for one image: logits `[Q,K+1]`, boxes `[Q,4]`, targets `[M]`/`[M,4]`.
    Cost combines negative class probability and L1 box distance. Real DETR uses
    the Hungarian algorithm and also generalized-IoU cost.
    """
    m, q = target_classes.numel(), class_logits.shape[0]
    if m > min(q, 8): raise ValueError("educational exhaustive matcher requires M <= min(Q, 8)")
    cost = -class_logits.softmax(-1)[:, target_classes] + torch.cdist(predicted_boxes, target_boxes, p=1)
    best_cost, best_queries = float("inf"), None
    for query_ids in permutations(range(q), m):
        value = sum(float(cost[query_ids[j], j]) for j in range(m))
        if value < best_cost: best_cost, best_queries = value, query_ids
    return torch.tensor(best_queries, dtype=torch.long), torch.arange(m)


def detr_loss(logits: Tensor, boxes: Tensor, target_classes: Tensor, target_boxes: Tensor) -> dict[str, Tensor]:
    """Match one image, mark unmatched queries no-object, then CE + matched L1."""
    query_ids, target_ids = tiny_bipartite_match(logits, boxes, target_classes, target_boxes)
    no_object_id = logits.shape[-1] - 1
    query_targets = torch.full((logits.shape[0],), no_object_id, dtype=torch.long)
    query_targets[query_ids] = target_classes[target_ids]
    classification = F.cross_entropy(logits, query_targets)
    box_l1 = F.l1_loss(boxes[query_ids], target_boxes[target_ids])
    return {"classification": classification, "box_l1": box_l1, "total": classification + 5*box_l1}


if __name__ == "__main__":
    torch.manual_seed(0)
    samples = [
        {"image": torch.randn(3, 40, 50), "boxes_xywh": torch.tensor([[5., 8., 10., 12.]]), "class_labels": torch.tensor([1])},
        {"image": torch.randn(3, 32, 44), "boxes_xywh": torch.tensor([[2., 3., 8., 9.], [20., 10., 12., 15.]]), "class_labels": torch.tensor([0, 2])},
    ]
    batch = collate_detection_batch(samples)
    print("pixel_values", tuple(batch["pixel_values"].shape)); print("pixel_mask  ", tuple(batch["pixel_mask"].shape))
    for i, label in enumerate(batch["labels"]): print(f"labels[{i}] classes={tuple(label['class_labels'].shape)} boxes={tuple(label['boxes'].shape)}")
    losses = detr_loss(torch.randn(4, 4), torch.rand(4, 4), batch["labels"][1]["class_labels"], batch["labels"][1]["boxes"])
    print({name: round(value.item(), 4) for name, value in losses.items()})
