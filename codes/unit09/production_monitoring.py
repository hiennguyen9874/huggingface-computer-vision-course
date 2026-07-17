"""Monitor latency, errors, confidence drift, and an A/B model rollout.

Run: uv run --extra cpu python codes/unit09/production_monitoring.py
Each request record contains scalar latency/error plus probabilities `[K]`.
Aggregates contain rates, percentiles, and a class-distribution drift score.
Do not log raw private images when aggregate telemetry is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RequestRecord:
    latency_ms: float
    probabilities: Tensor  # float32 [K], non-negative and summing to approximately 1
    had_error: bool = False


@dataclass(frozen=True)
class HealthReport:
    requests: int
    error_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_confidence: float
    distribution_l1_drift: float


def nearest_rank_percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, int(len(ordered) * fraction + 0.9999) - 1)
    return ordered[index]


def summarize(records: list[RequestRecord], baseline_distribution: Tensor) -> HealthReport:
    """Aggregate bounded telemetry and compare output distribution to baseline.

    L1 drift ranges from 0 (identical class distribution) to 2 (disjoint). It is
    a warning signal, not proof of accuracy loss; investigate with labeled data.
    """
    if not records:
        raise ValueError("at least one request record is required")
    class_count = baseline_distribution.numel()
    if baseline_distribution.ndim != 1 or not torch.isclose(
            baseline_distribution.sum(), torch.tensor(1.0), atol=1e-5):
        raise ValueError("baseline_distribution must be probabilities [K] summing to 1")
    for record in records:
        probabilities = record.probabilities
        if probabilities.shape != (class_count,) or not torch.isclose(
                probabilities.sum(), torch.tensor(1.0), atol=1e-5):
            raise ValueError(f"every probability vector must have shape [{class_count}] and sum to 1")
        if record.latency_ms < 0:
            raise ValueError("latency cannot be negative")

    latencies = [record.latency_ms for record in records]
    probability_batch = torch.stack([record.probabilities for record in records]) # [N,K]
    production_distribution = probability_batch.mean(dim=0)                       # [K]
    return HealthReport(
        requests=len(records),
        error_rate=sum(record.had_error for record in records) / len(records),
        mean_latency_ms=sum(latencies) / len(latencies),
        p95_latency_ms=nearest_rank_percentile(latencies, 0.95),
        mean_confidence=probability_batch.max(dim=1).values.mean().item(),
        distribution_l1_drift=(production_distribution - baseline_distribution).abs().sum().item(),
    )


def compare_ab(control: HealthReport, candidate: HealthReport) -> dict[str, float]:
    """Return candidate-minus-control deltas; lower latency/error are better."""
    return {
        "p95_latency_delta_ms": candidate.p95_latency_ms - control.p95_latency_ms,
        "error_rate_delta": candidate.error_rate - control.error_rate,
        "confidence_delta": candidate.mean_confidence - control.mean_confidence,
        "drift_delta": candidate.distribution_l1_drift - control.distribution_l1_drift,
    }


def make_records(latencies: list[float], logits: Tensor, errors: set[int]) -> list[RequestRecord]:
    """Turn model logits `[N,K]` and request telemetry into validated records."""
    if logits.ndim != 2 or logits.shape[0] != len(latencies):
        raise ValueError("logits must be [N,K] with one latency per request")
    probabilities = logits.softmax(dim=1)
    return [RequestRecord(latency, probabilities[index], index in errors)
            for index, latency in enumerate(latencies)]


if __name__ == "__main__":
    baseline = torch.tensor([0.40, 0.35, 0.25], dtype=torch.float32)
    control_logits = torch.tensor([[2.0, 1.0, 0.0], [0.5, 1.5, 0.2],
                                   [0.1, 0.3, 1.7], [1.6, 0.8, 0.2]])
    candidate_logits = torch.tensor([[2.3, 0.8, 0.0], [0.3, 1.8, 0.1],
                                     [0.2, 0.2, 1.9], [1.9, 0.6, 0.1]])
    control = summarize(make_records([42, 51, 47, 120], control_logits, {3}), baseline)
    candidate = summarize(make_records([35, 39, 37, 48], candidate_logits, set()), baseline)

    print(f"control logits   shape={tuple(control_logits.shape)}, dtype={control_logits.dtype}")
    print(f"candidate logits shape={tuple(candidate_logits.shape)}, dtype={candidate_logits.dtype}")
    print(f"control report   {control}")
    print(f"candidate report {candidate}")
    print(f"A/B deltas       {compare_ab(control, candidate)}")
    print("rollout decision requires product quality metrics and enough traffic; telemetry alone is insufficient")
