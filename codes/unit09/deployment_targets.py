"""Choose a deployment runtime from explicit product/hardware constraints.

Run: uv run --extra cpu python codes/unit09/deployment_targets.py
This module models the chapter's hardware decision; it does not claim to run or
benchmark SDKs that require target-specific packages, drivers, or devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Target(str, Enum):
    NVIDIA_GPU = "nvidia_gpu"
    INTEL = "intel_cpu_gpu_vpu"
    MOBILE = "mobile"
    EDGE_TPU = "edge_tpu"
    PORTABLE = "cross_platform"
    CLOUD_MANAGED = "cloud_managed"


@dataclass(frozen=True)
class Requirements:
    target: Target
    offline: bool
    latency_budget_ms: float
    memory_budget_mb: int


@dataclass(frozen=True)
class DeploymentPlan:
    runtime: str
    artifact: str
    precision: str
    reason: str


def choose_plan(requirements: Requirements) -> DeploymentPlan:
    """Return a transparent baseline plan; benchmark it on the real target.

    Runtime names encode chapter guidance, not guaranteed performance. Operator
    support, accuracy parity, latency, and memory must be measured after export.
    """
    if requirements.latency_budget_ms <= 0 or requirements.memory_budget_mb <= 0:
        raise ValueError("latency and memory budgets must be positive")
    if requirements.target is Target.NVIDIA_GPU:
        return DeploymentPlan("TensorRT", "ONNX -> TensorRT engine", "FP16/INT8",
                              "NVIDIA GPU kernels and engine optimization")
    if requirements.target is Target.INTEL:
        return DeploymentPlan("OpenVINO", "OpenVINO IR", "FP16/INT8",
                              "optimized for Intel CPU/GPU/VPU")
    if requirements.target is Target.EDGE_TPU:
        return DeploymentPlan("Edge TPU runtime", "compiled TFLite", "INT8",
                              "Edge TPU requires supported quantized operators")
    if requirements.target is Target.MOBILE:
        return DeploymentPlan("TFLite/Core ML/ORT Mobile", "mobile package", "FP16/INT8",
                              "small startup, memory, and battery footprint")
    if requirements.target is Target.CLOUD_MANAGED:
        return DeploymentPlan("HF Endpoint/managed service", "versioned model package", "FP16",
                              "managed scaling, rollout, and monitoring")
    return DeploymentPlan("ONNX Runtime", "ONNX", "FP32/FP16/INT8",
                          "portable runtime across operating systems and languages")


if __name__ == "__main__":
    examples = [
        Requirements(Target.NVIDIA_GPU, offline=False, latency_budget_ms=20, memory_budget_mb=2048),
        Requirements(Target.EDGE_TPU, offline=True, latency_budget_ms=33, memory_budget_mb=128),
        Requirements(Target.PORTABLE, offline=False, latency_budget_ms=100, memory_budget_mb=512),
    ]
    for requirements in examples:
        plan = choose_plan(requirements)
        print(f"target={requirements.target.value:18} offline={requirements.offline!s:5} "
              f"budget={requirements.latency_budget_ms:g}ms/{requirements.memory_budget_mb}MB")
        print(f"  runtime={plan.runtime}; artifact={plan.artifact}; precision={plan.precision}")
        print(f"  verify: output parity, P95 latency, peak memory; reason={plan.reason}")
