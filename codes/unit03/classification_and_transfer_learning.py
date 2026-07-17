"""Multi-class/multi-label objectives and ViT transfer-learning modes.

Run: uv run python codes/unit03/classification_and_transfer_learning.py
"""

from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from vit import VisionTransformer


def multiclass_objective(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """One class per sample.

    logits: float `[N, K]`; labels: int64 `[N]` in `[0, K)`.
    Returns scalar cross-entropy loss and probabilities `[N, K]` whose rows sum to 1.
    """
    return F.cross_entropy(logits, labels), logits.softmax(dim=-1)


def multilabel_objective(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Any number of independent labels per sample.

    logits/targets: `[N, K]`; targets contain 0/1 floats. Returns scalar
    binary-cross-entropy loss and independent probabilities `[N, K]`.
    """
    return F.binary_cross_entropy_with_logits(logits, targets), logits.sigmoid()


def configure_transfer_learning(model: VisionTransformer, train_backbone: bool) -> None:
    """Choose feature extraction (`False`) or full fine-tuning (`True`).

    The newly replaced classification head always remains trainable. Full
    fine-tuning normally uses a smaller learning rate than head-only training.
    """
    for parameter in model.parameters():
        parameter.requires_grad = train_backbone
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = VisionTransformer(num_classes=3)
    images = torch.randn(2, 3, 32, 32)  # [N=2, RGB, H=32, W=32]
    logits = model(images)              # [2, K=3], no activation before a *WithLogits loss

    mc_loss, mc_probs = multiclass_objective(logits, torch.tensor([0, 2]))
    ml_targets = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
    ml_loss, ml_probs = multilabel_objective(logits, ml_targets)
    print(f"images                 {tuple(images.shape)}")
    print(f"logits                 {tuple(logits.shape)}")
    print(f"multiclass probabilities {tuple(mc_probs.shape)}, row sums={mc_probs.sum(-1).tolist()}")
    print(f"multilabel probabilities {tuple(ml_probs.shape)}, independent scores={ml_probs[0].tolist()}")
    print(f"losses: multiclass={mc_loss.item():.4f}, multilabel={ml_loss.item():.4f}")

    configure_transfer_learning(model, train_backbone=False)
    print(f"feature extraction trainable parameters: {trainable_parameter_count(model):,}")
    configure_transfer_learning(model, train_backbone=True)
    print(f"full fine-tuning trainable parameters:    {trainable_parameter_count(model):,}")
