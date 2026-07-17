# Unit 9 — Model optimization and deployment

Small, executable lessons matching `chapters-vi/unit09.md` and
`chapters-en/unit9/`. Each file owns one concept, states its input/output
contract, and prints the shape and dtype at each meaningful step.

| File | Concept | Main contract |
|---|---|---|
| `pruning.py` | unstructured and structured pruning | dense weights -> sparse masks/weights |
| `quantization.py` | post-training dynamic INT8 quantization | FP32 classifier -> quantized classifier |
| `knowledge_distillation.py` | teacher/student training | teacher + student logits `[N,K]` -> scalar loss |
| `low_rank_approximation.py` | truncated-SVD compression | weight `[O,I]` -> factors `[R,I]`, `[O,R]` |
| `model_benchmarking.py` | size, latency, throughput, parity | model + image batch -> deployment metrics |
| `serialization_and_packaging.py` | safe model artifact packaging | state dict + metadata -> reproducible package |
| `inference_pipeline.py` | preprocessing, inference, postprocessing | uint8 HWC image -> JSON-ready prediction |
| `production_monitoring.py` | latency percentiles, drift, A/B comparison | request records -> health report |
| `deployment_targets.py` | target/runtime selection | deployment requirements -> runtime plan |

The neural networks are intentionally small teaching models with random weights.
They demonstrate mechanics and contracts; they are not trained production
checkpoints. Hardware SDKs named by the chapter (TensorRT, OpenVINO, Edge TPU,
ONNX Runtime, and Optimum) are target-specific and are not project dependencies.
`deployment_targets.py` explains when they belong in a real deployment without
requiring unavailable hardware or pretending to benchmark it.

Notation: `N` batch size, `C` channels, `H/W` image height/width, `F` feature
width, `K` classes, `I/O` linear input/output width, and `R` retained rank.

Run one lesson from the repository root:

```bash
uv run --extra cpu python codes/unit09/pruning.py
```

Run all lessons:

```bash
for file in codes/unit09/*.py; do uv run --extra cpu python "$file"; done
```

All demos are deterministic, CPU-compatible, and require no network access.
Temporary model packages are automatically removed after their demonstrations.
