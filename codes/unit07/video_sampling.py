"""Video tensor layouts and deterministic uniform frame sampling.

Run: uv run python codes/unit07/video_sampling.py
"""
from __future__ import annotations
import torch
from torch import Tensor


def uniform_sample_frames(video: Tensor, num_frames: int) -> tuple[Tensor, Tensor]:
    """Sample evenly spaced frames.

    Input: `video` float/uint8 `[T,H,W,C]`; `num_frames` in `[1,T]`.
    Output: sampled video `[S,H,W,C]` and int64 source indices `[S]`, where
    `S=num_frames`. Endpoint inclusion gives coverage of the whole clip.
    """
    if video.ndim != 4:
        raise ValueError(f"expected video [T,H,W,C], got {tuple(video.shape)}")
    total = video.shape[0]
    if not 1 <= num_frames <= total:
        raise ValueError(f"num_frames must be in [1,{total}], got {num_frames}")
    indices = torch.linspace(0, total - 1, num_frames, device=video.device).round().long()
    return video.index_select(0, indices), indices


def to_model_layout(video: Tensor) -> Tensor:
    """Convert one `[T,H,W,C]` clip to float `[N=1,C,T,H,W]` for Conv3d."""
    if video.ndim != 4:
        raise ValueError(f"expected [T,H,W,C], got {tuple(video.shape)}")
    return video.permute(3, 0, 1, 2).unsqueeze(0).float()


if __name__ == "__main__":
    video = torch.randint(0, 256, (20, 32, 48, 3), dtype=torch.uint8)
    sampled, indices = uniform_sample_frames(video, 6)
    model_input = to_model_layout(sampled) / 255.0
    print(f"decoded video  shape={tuple(video.shape)}, dtype={video.dtype}  [T,H,W,C]")
    print(f"indices        shape={tuple(indices.shape)}, dtype={indices.dtype}, values={indices.tolist()}")
    print(f"sampled clip   shape={tuple(sampled.shape)}, dtype={sampled.dtype}  [S,H,W,C]")
    print(f"model input    shape={tuple(model_input.shape)}, dtype={model_input.dtype}  [N,C,T,H,W]")
