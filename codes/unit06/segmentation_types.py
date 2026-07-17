"""Concrete output contracts for semantic, instance, and panoptic segmentation.

Run: uv run python codes/unit06/segmentation_types.py
This file models task outputs rather than a neural network. It shows why one
semantic map cannot distinguish two objects belonging to the same class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class InstancePrediction:
    """One detected object: scalar class/score plus a boolean mask `[H,W]`."""

    class_id: int
    score: float
    mask: Tensor


def semantic_from_instances(instances: list[InstancePrediction], height: int, width: int) -> Tensor:
    """Merge object masks into one integer class map `[H,W]`; 0 is background."""
    semantic = torch.zeros(height, width, dtype=torch.long)
    # Low-confidence masks are painted first, so stronger overlapping masks win.
    for instance in sorted(instances, key=lambda item: item.score):
        if instance.mask.shape != (height, width) or instance.mask.dtype != torch.bool:
            raise ValueError(f"expected boolean mask [{height}, {width}], got {instance.mask.shape}")
        semantic[instance.mask] = instance.class_id
    return semantic


def panoptic_from_instances(
    instances: list[InstancePrediction], height: int, width: int
) -> tuple[Tensor, Tensor]:
    """Return class and instance-ID maps, each integer `[H,W]`.

    Instance ID 0 means stuff/background. Positive IDs distinguish countable
    objects even when their class IDs are equal.
    """
    class_map = torch.zeros(height, width, dtype=torch.long)
    instance_map = torch.zeros(height, width, dtype=torch.long)
    for instance_id, instance in enumerate(sorted(instances, key=lambda item: item.score), start=1):
        class_map[instance.mask] = instance.class_id
        instance_map[instance.mask] = instance_id
    return class_map, instance_map


if __name__ == "__main__":
    height, width = 8, 10
    cat_1 = torch.zeros(height, width, dtype=torch.bool)
    cat_2 = torch.zeros(height, width, dtype=torch.bool)
    cat_1[1:5, 1:4] = True
    cat_2[2:7, 6:9] = True
    instances = [
        InstancePrediction(class_id=1, score=0.98, mask=cat_1),
        InstancePrediction(class_id=1, score=0.96, mask=cat_2),
    ]
    semantic = semantic_from_instances(instances, height, width)
    panoptic_classes, panoptic_instances = panoptic_from_instances(instances, height, width)
    stacked_instance_masks = torch.stack([item.mask for item in instances])  # [objects,H,W]
    print(f"semantic class map   shape={tuple(semantic.shape)}, dtype={semantic.dtype}")
    print(f"instance masks       shape={tuple(stacked_instance_masks.shape)}, dtype={stacked_instance_masks.dtype}")
    print(f"panoptic class map   shape={tuple(panoptic_classes.shape)}, dtype={panoptic_classes.dtype}")
    print(f"panoptic instance map shape={tuple(panoptic_instances.shape)}, IDs={panoptic_instances.unique().tolist()}")
    print("Both cats have semantic class 1, but panoptic instance IDs are 1 and 2.")
