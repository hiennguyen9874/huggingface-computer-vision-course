"""Tiny Two-Stream action network for RGB appearance and frame motion.

Run: uv run python codes/unit07/two_stream_network.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class FrameStream(nn.Module):
    """Apply one shared 2D CNN to frames `[N,C,T,H,W]`, then average time/space."""
    def __init__(self, channels: int, feature_dim: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(channels, 16, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(16, feature_dim, 3, 2, 1), nn.ReLU())

    def forward(self, clip: Tensor) -> Tensor:
        n, c, t, h, w = clip.shape
        frames = clip.transpose(1, 2).reshape(n * t, c, h, w)  # [N*T,C,H,W]
        maps = self.cnn(frames)                                 # [N*T,D,H/4,W/4]
        return maps.mean((2, 3)).reshape(n, t, -1).mean(1)      # [N,D]


class TwoStreamNetwork(nn.Module):
    """Fuse RGB and motion features; inputs `[N,3,T,H,W]`, output `[N,K]` logits.

    Motion uses adjacent RGB differences as a cheap differentiable optical-flow
    proxy. A production system can pass true optical flow to the temporal stream.
    """
    def __init__(self, num_classes: int = 5, feature_dim: int = 32) -> None:
        super().__init__()
        self.spatial = FrameStream(3, feature_dim)
        self.temporal = FrameStream(3, feature_dim)
        self.classifier = nn.Linear(2 * feature_dim, num_classes)

    def forward_with_shapes(self, video: Tensor) -> dict[str, Tensor]:
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] < 2:
            raise ValueError(f"expected [N,3,T>=2,H,W], got {tuple(video.shape)}")
        motion = video[:, :, 1:] - video[:, :, :-1]       # [N,3,T-1,H,W]
        appearance = self.spatial(video)                   # [N,D]
        motion_features = self.temporal(motion)            # [N,D]
        fused = torch.cat((appearance, motion_features), 1)# [N,2D]
        return {"video": video, "motion": motion, "appearance": appearance,
                "motion_features": motion_features, "fused": fused,
                "logits": self.classifier(fused)}

    def forward(self, video: Tensor) -> Tensor:
        return self.forward_with_shapes(video)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    trace = TwoStreamNetwork().forward_with_shapes(torch.randn(2, 3, 6, 32, 32))
    for name, value in trace.items(): print(f"{name:17} shape={tuple(value.shape)}, dtype={value.dtype}")
