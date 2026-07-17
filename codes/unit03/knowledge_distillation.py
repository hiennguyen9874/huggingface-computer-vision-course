"""Knowledge distillation: a small student learns labels and teacher soft targets.

Run: uv run python codes/unit03/knowledge_distillation.py
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DistillationLoss:
    """Named scalar outputs make the training contract explicit."""
    total: Tensor
    hard_label: Tensor
    soft_teacher: Tensor


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor, labels: Tensor,
                       temperature: float = 2.0, alpha: float = 0.5) -> DistillationLoss:
    """Combine ground-truth cross entropy and teacher-student KL divergence.

    student_logits/teacher_logits: `[N,K]`; labels: int64 `[N]`.
    alpha weights the hard-label term. T>1 softens class distributions and
    reveals class similarity. Multiplication by T^2 preserves gradient scale.
    Teacher logits must be detached/no-grad during student training.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical [N,K] shape")
    hard = F.cross_entropy(student_logits, labels)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    soft = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * temperature**2
    return DistillationLoss(alpha*hard + (1-alpha)*soft, hard, soft)


class ImageClassifier(nn.Module):
    """Width-configurable CNN used to make teacher/student size differences visible."""
    def __init__(self, width: int, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, width, 3, 2, 1), nn.ReLU(),
                                      nn.Conv2d(width, 2*width, 3, 2, 1), nn.ReLU())
        self.head = nn.Linear(2*width, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)             # [N,2*width,H/4,W/4]
        return self.head(features.mean((2, 3)))      # [N,K]


def parameters(model: nn.Module) -> int: return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0); teacher, student = ImageClassifier(64), ImageClassifier(16)
    images, labels = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])
    with torch.no_grad(): teacher_logits = teacher(images)  # teacher is fixed
    student_logits = student(images)
    losses = distillation_loss(student_logits, teacher_logits, labels)
    losses.total.backward()  # only student receives gradients
    print(f"images          {tuple(images.shape)}")
    print(f"teacher logits  {tuple(teacher_logits.shape)}; parameters={parameters(teacher):,}")
    print(f"student logits  {tuple(student_logits.shape)}; parameters={parameters(student):,}")
    print(f"hard CE={losses.hard_label.item():.4f}, soft KL={losses.soft_teacher.item():.4f}, total={losses.total.item():.4f}")
    print("student head gradient", tuple(student.head.weight.grad.shape))
