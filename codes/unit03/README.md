# Unit 3 code — Vision Transformers

These standalone PyTorch lessons implement the model mechanics described in `chapters-vi/unit03.md` and `chapters-en/unit3/vision-transformers/`. They use small random tensors so every example runs on CPU without datasets, network access, pretrained weights, or hidden preprocessing.

## Design

Each file owns one cohesive model or training boundary, documents tensor type/shape/input/output, exposes a `forward_with_shapes` trace where useful, and has a deterministic `__main__` demonstration. The implementations are intentionally much smaller than research/production checkpoints: they teach data flow and contracts, not checkpoint-compatible accuracy.

| File | Chapter model/section | What the runnable trace demonstrates |
| --- | --- | --- |
| `vit.py` | Vision Transformer | convolutional patch projection, class token, positional embedding, encoder, logits |
| `classification_and_transfer_learning.py` | transfer learning; multi-class vs multi-label | frozen backbone vs full fine-tuning, softmax/CE vs sigmoid/BCE |
| `swin_transformer.py` | Swin, Swin V2 foundation | window partition/reverse, regular and shifted-window attention, hierarchical patch merging |
| `cvt.py` | Convolutional Vision Transformer | overlapping convolutional tokens, depthwise convolutional Q/K/V, no positional embedding |
| `dinat.py` | DiNAT | local and dilated neighborhood attention with linear-in-image-size neighborhood cost |
| `mobilevit_v2.py` | MobileViT v2 | local CNN features, unfold/fold, separable self-attention, residual fusion |
| `detr.py` | DETR | CNN feature sequence, encoder memory, learned object queries, class/no-object and box heads |
| `detection_training.py` | fine-tuning object detection | COCO box conversion, synchronized flip, variable-size padding/masks, matching and set loss |
| `maskformer.py` | transformer segmentation / MaskFormer | pixel module, segment queries, class head, mask-embedding × pixel-embedding masks |
| `oneformer.py` | OneFormer | semantic/instance/panoptic task conditioning and query-text contrastive loss |
| `knowledge_distillation.py` | teacher-student distillation | hard-label CE, temperature-scaled KL, fixed teacher, smaller student |

The chapter's Deformable DETR and Conditional DETR are evolutions of the DETR attention/query mechanism rather than separate introductory implementations. Swin V2 changes training stability and relative-position formulation; SwinIR/Swin2SR apply Swin blocks to restoration. Their shared core is made executable in `swin_transformer.py` while the chapter text remains the source for those production extensions.

## Run

From the repository root:

```bash
uv run python codes/unit03/vit.py
uv run python codes/unit03/classification_and_transfer_learning.py
uv run python codes/unit03/swin_transformer.py
uv run python codes/unit03/cvt.py
uv run python codes/unit03/dinat.py
uv run python codes/unit03/mobilevit_v2.py
uv run python codes/unit03/detr.py
uv run python codes/unit03/detection_training.py
uv run python codes/unit03/maskformer.py
uv run python codes/unit03/oneformer.py
uv run python codes/unit03/knowledge_distillation.py
```

`classification_and_transfer_learning.py` imports the local `vit.py`; running it by file path as shown makes `codes/unit03` available on Python's import path.

## Shape notation

- `N`: batch size
- `C`: image/feature channels
- `H, W`: spatial height and width
- `P`: patch size or pixels per patch (specified locally)
- `L`: token/patch count
- `D` or `E`: embedding dimension
- `Q`: object or segment query count
- `K`: number of task classes

Logits are intentionally returned without activation so numerically stable `CrossEntropyLoss` / `BCEWithLogitsLoss` can consume them directly. Bounding-box formats are named at every boundary (`xywh`, `cxcywh`, or `xyxy`) to avoid silent label corruption.
