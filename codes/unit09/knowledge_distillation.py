"""Knowledge distillation trains a small student from labels and a teacher.

Run: uv run --extra cpu python codes/unit09/knowledge_distillation.py
Inputs: images `[N,3,H,W]`, labels int64 `[N]`, logits `[N,K]`.
Output: scalar loss; gradients are produced only for the student.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ImageClassifier(nn.Module):
    """Width-configurable CNN used to expose teacher/student capacity."""

    def __init__(self, width: int, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), # -> [N,width,H/2,W/2]
            nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
            nn.ReLU(),                                  # -> [N,2*width,H/4,W/4]
        )
        self.head = nn.Linear(2 * width, num_classes)   # [N,2*width] -> [N,K]

    def forward(self, images: Tensor) -> Tensor:
        maps = self.features(images)
        pooled = maps.mean(dim=(2, 3))                  # global average -> [N,2*width]
        return self.head(pooled)


@dataclass(frozen=True)
class DistillationLoss:
    total: Tensor         # scalar optimized by the student
    hard_label: Tensor    # scalar cross-entropy against labels
    soft_teacher: Tensor  # scalar KL divergence against teacher probabilities


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor,
                       labels: Tensor, temperature: float = 4.0,
                       hard_weight: float = 0.5) -> DistillationLoss:
    """Blend ground-truth CE with temperature-scaled teacher/student KL.

    Dividing logits by `temperature > 1` reveals relative probabilities among
    non-winning classes. Multiplying KL by T² keeps gradient scale comparable.
    Detaching teacher logits prevents accidental teacher optimization.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must both have shape [N,K]")
    if labels.shape != student_logits.shape[:1] or labels.dtype != torch.int64:
        raise ValueError("labels must be int64 with shape [N]")
    if temperature <= 0 or not 0 <= hard_weight <= 1:
        raise ValueError("temperature must be positive and hard_weight in [0,1]")

    hard = F.cross_entropy(student_logits, labels)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    soft = F.kl_div(student_log_probs, teacher_probs,
                    reduction="batchmean") * temperature**2
    total = hard_weight * hard + (1 - hard_weight) * soft
    return DistillationLoss(total, hard, soft)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    teacher = ImageClassifier(width=64).eval()
    student = ImageClassifier(width=16).train()
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

    with torch.no_grad():
        teacher_logits = teacher(images)  # fixed teacher: [N,K]
    student_logits = student(images)      # trainable student: [N,K]
    losses = distillation_loss(student_logits, teacher_logits, labels)
    losses.total.backward()

    print(f"images          shape={tuple(images.shape)}, dtype={images.dtype}")
    print(f"labels          shape={tuple(labels.shape)}, dtype={labels.dtype}")
    print(f"teacher logits  shape={tuple(teacher_logits.shape)}, params={parameter_count(teacher):,}")
    print(f"student logits  shape={tuple(student_logits.shape)}, params={parameter_count(student):,}")
    print(f"losses          hard={losses.hard_label.item():.4f}, "
          f"soft={losses.soft_teacher.item():.4f}, total={losses.total.item():.4f}")
    print(f"student gradient shape={tuple(student.head.weight.grad.shape)}; "
          f"teacher gradient={teacher.head.weight.grad}")
