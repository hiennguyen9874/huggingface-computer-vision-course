"""Reference-free and paired metrics used for generated images.

The FID/IS/CLIP functions consume already-extracted model outputs so this lesson
runs offline. In production, obtain them from Inception-v3 or CLIP.
Run: uv run python codes/unit05/generative_metrics.py
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def psnr(reference: Tensor, generated: Tensor, data_range: float = 1.0) -> Tensor:
    """Peak signal-to-noise ratio for equal-shaped `[N, C, H, W]` images.

    Input values should lie in an interval of width ``data_range``. Output is one
    scalar in decibels; higher is better and identical inputs return infinity.
    """
    if reference.shape != generated.shape:
        raise ValueError(f"expected equal image shapes, got {reference.shape} and {generated.shape}")
    mse = F.mse_loss(generated, reference)
    return 10 * torch.log10(torch.as_tensor(data_range**2, device=mse.device) / mse)


def ssim(reference: Tensor, generated: Tensor, data_range: float = 1.0) -> Tensor:
    """Global SSIM over `[N, C, H, W]`; output scalar is normally in `[-1, 1]`.

    This clear global form demonstrates luminance, contrast, and structure. Real
    evaluations commonly use a local Gaussian window implementation.
    """
    if reference.shape != generated.shape or reference.ndim != 4:
        raise ValueError("SSIM expects equal [N, C, H, W] tensors")
    dims = (-2, -1)
    mu_x, mu_y = reference.mean(dims), generated.mean(dims)  # [N, C]
    var_x = reference.var(dims, correction=0)  # [N, C]
    var_y = generated.var(dims, correction=0)
    covariance = ((reference - mu_x[..., None, None]) * (generated - mu_y[..., None, None])).mean(dims)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return score.mean()


def _covariance(features: Tensor) -> Tensor:
    centered = features - features.mean(0)
    return centered.T @ centered / (features.shape[0] - 1)


def _symmetric_matrix_sqrt(matrix: Tensor) -> Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh((matrix + matrix.T) / 2)
    return (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T


def fid(real_features: Tensor, fake_features: Tensor) -> Tensor:
    """FID from Inception-like features `[Nr, D]` and `[Nf, D]`; lower is better."""
    if real_features.ndim != 2 or fake_features.ndim != 2 or real_features.shape[1] != fake_features.shape[1]:
        raise ValueError("FID expects [samples, feature_dim] tensors with the same feature_dim")
    if min(real_features.shape[0], fake_features.shape[0]) < 2:
        raise ValueError("FID covariance requires at least two samples per set")
    mu_r, mu_f = real_features.mean(0), fake_features.mean(0)  # [D]
    cov_r, cov_f = _covariance(real_features), _covariance(fake_features)  # [D, D]
    # This symmetric construction has the same eigenvalues as cov_r @ cov_f,
    # while avoiding a non-symmetric matrix square root.
    sqrt_r = _symmetric_matrix_sqrt(cov_r)
    covariance_mean = _symmetric_matrix_sqrt(sqrt_r @ cov_f @ sqrt_r)
    return (mu_r - mu_f).square().sum() + torch.trace(cov_r + cov_f - 2 * covariance_mean)


def inception_score(class_logits: Tensor) -> Tensor:
    """IS from classifier logits `[N, classes]`; output scalar, higher is better."""
    probabilities = class_logits.softmax(-1)  # p(y|x), [N, K]
    marginal = probabilities.mean(0, keepdim=True)  # p(y), [1, K]
    kl = (probabilities * (probabilities.clamp_min(1e-8).log() - marginal.clamp_min(1e-8).log())).sum(-1)
    return kl.mean().exp()


def clip_score(image_embeddings: Tensor, text_embeddings: Tensor) -> Tensor:
    """Mean paired cosine score for `[N, D]` image/text embeddings, scaled 0..100."""
    if image_embeddings.shape != text_embeddings.shape:
        raise ValueError("CLIP score expects paired [N, D] embeddings of equal shape")
    cosine = F.cosine_similarity(image_embeddings, text_embeddings, dim=-1)  # [N]
    return 100 * cosine.clamp_min(0).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    real = torch.rand(8, 3, 16, 16)
    fake = (real + 0.05 * torch.randn_like(real)).clamp(0, 1)
    real_features = torch.randn(32, 16)
    fake_features = real_features + 0.2 * torch.randn_like(real_features)
    logits = torch.randn(32, 10)
    image_embeddings, text_embeddings = torch.randn(8, 32), torch.randn(8, 32)
    print(f"images: real={tuple(real.shape)}, generated={tuple(fake.shape)}")
    print(f"PSNR={psnr(real, fake):.3f} dB (higher is better)")
    print(f"SSIM={ssim(real, fake):.3f} (higher is better)")
    print(f"FID={fid(real_features, fake_features):.3f} (lower is better)")
    print(f"IS={inception_score(logits):.3f} (higher is better)")
    print(f"CLIP score={clip_score(image_embeddings, text_embeddings):.3f} (higher is better)")
