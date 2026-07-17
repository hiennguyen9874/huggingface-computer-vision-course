# Unit 6 — Basic computer vision tasks

Small, executable PyTorch lessons matching `chapters-vi/unit06.md` and
`chapters-en/unit6/basic-cv-tasks/`. The neural networks are compact teaching
implementations with random weights; they expose the contracts and tensor flow
of each task without downloading pretrained checkpoints.

| File | Concept | Main contract |
|---|---|---|
| `image_classification.py` | CNN image classifier | image `[N,C,H,W]` -> class logits `[N,K]` |
| `object_detection.py` | DETR-style set prediction | image -> class logits and normalized boxes per query |
| `detection_metrics.py` | box IoU, precision/recall, AP, mAP | detections + ground truth -> scalar metrics |
| `segmentation_types.py` | semantic, instance, panoptic outputs | pixel/instance masks -> task-specific maps |
| `unet.py` | U-Net encoder, skips, decoder | image -> per-pixel class logits |
| `promptable_segmentation.py` | SAM-style point/box prompting | image + prompt -> candidate mask logits |
| `segmentation_metrics.py` | mask IoU, pixel accuracy, Dice | predicted + target masks -> scalar metrics |

The DETR-style and SAM-style files explain the architecture ideas used by the
models named in the chapter; they are deliberately much smaller and are not
compatible with pretrained DETR or SAM weights.

Run one lesson:

```bash
uv run python codes/unit06/unet.py
```

Run all lessons:

```bash
for file in codes/unit06/*.py; do uv run python "$file"; done
```

Notation: `N` batch size, `C` channels, `H/W` image height/width, `K` classes,
`D` feature width, `Q` object queries, and `M` mask candidates.
