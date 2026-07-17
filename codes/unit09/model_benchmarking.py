"""Benchmark model size, latency, throughput, and output quality together.

Run: uv run --extra cpu python codes/unit09/model_benchmarking.py
Input images `[N,3,H,W]` and labels `[N]`; output logits `[N,K]` and metrics.
Timing tiny synthetic models is educational only—benchmark production models on
the actual target hardware with realistic inputs, concurrency, and warm-up.
"""

from __future__ import annotations

import io
import statistics
import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class DeployableCNN(nn.Module):
    """Width-configurable classifier for comparing resource/latency trade-offs."""

    def __init__(self, width: int, num_classes: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),  # -> [N,width,H,W]
            nn.MaxPool2d(2),                               # -> [N,width,H/2,W/2]
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                       # -> [N,2*width,1,1]
        )
        self.head = nn.Linear(2 * width, num_classes)      # -> [N,K]

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.features(images).flatten(1))


@dataclass(frozen=True)
class Benchmark:
    parameters: int
    artifact_bytes: int
    median_latency_ms: float
    p95_latency_ms: float
    throughput_images_s: float
    top1_accuracy: float


def state_dict_size(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; input values must already be sorted."""
    index = max(0, int(len(sorted_values) * fraction + 0.9999) - 1)
    return sorted_values[index]


def benchmark(model: nn.Module, images: Tensor, labels: Tensor,
              warmup: int = 3, iterations: int = 20) -> tuple[Tensor, Benchmark]:
    """Measure one fixed batch under inference mode and return its final logits."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(images)
        latencies: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            logits = model(images)                         # [N,K]
            latencies.append((time.perf_counter() - started) * 1000)

    latencies.sort()
    predictions = logits.argmax(dim=1)                     # int64 [N]
    accuracy = (predictions == labels).float().mean().item()
    median_ms = statistics.median(latencies)
    metrics = Benchmark(
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        artifact_bytes=state_dict_size(model),
        median_latency_ms=median_ms,
        p95_latency_ms=percentile(latencies, 0.95),
        throughput_images_s=images.shape[0] / (median_ms / 1000),
        top1_accuracy=accuracy,
    )
    return logits, metrics


if __name__ == "__main__":
    torch.manual_seed(0)
    # One CPU thread reduces timing noise for this small educational workload.
    torch.set_num_threads(1)
    images = torch.randn(8, 3, 64, 64, dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.int64)
    models = {"compact": DeployableCNN(width=8), "wide": DeployableCNN(width=32)}

    print(f"images shape={tuple(images.shape)}, dtype={images.dtype}")
    print(f"labels shape={tuple(labels.shape)}, dtype={labels.dtype}")
    for name, model in models.items():
        logits, result = benchmark(model, images, labels)
        print(f"{name:7} logits={tuple(logits.shape)}, params={result.parameters:,}, "
              f"size={result.artifact_bytes / 1024:.1f}KiB")
        print(f"        median={result.median_latency_ms:.3f}ms, "
              f"p95={result.p95_latency_ms:.3f}ms, "
              f"throughput={result.throughput_images_s:.1f} images/s, "
              f"demo accuracy={result.top1_accuracy:.1%}")
