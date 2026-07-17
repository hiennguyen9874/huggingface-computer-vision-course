# Unit 2 code

These scripts turn the Unit 2 chapters into small, executable PyTorch lessons. Each file owns one concept/model, documents its tensor contract, and has a `__main__` demo that prints the shape after each meaningful step.

## Modules

| File | Chapter concept | Main output |
| --- | --- | --- |
| `convolution_and_pooling.py` | convolution, padding, stride, feature maps, pooling | a manually computed feature map and pooled tensor |
| `basic_cnn.py` | convolution -> ReLU -> pooling -> classifier | logits of shape `[batch, 10]` |
| `transfer_learning.py` | frozen backbone, new head, fine-tuning last block | logits from a ResNet-18 transfer model |
| `vgg.py` | VGG's repeated `3x3` convolutions and pooling | logits from a VGG-16-style classifier |
| `googlenet.py` | Inception's parallel multi-scale branches and auxiliary head | main and auxiliary logits |
| `resnet.py` | residual learning `F(x) + x` and projection shortcut | logits from a compact ResNet-18-style model |
| `mobilenet.py` | depthwise + pointwise convolution | logits and parameter-count comparison |
| `yolo.py` | YOLOv1 grid output, box decoding, IoU, NMS | final detections after post-processing |
| `convnext.py` | patchify stem, depthwise large kernel, LayerNorm, GELU | logits from a compact ConvNeXt |

The model implementations are deliberately smaller than production checkpoints. Their purpose is to expose shape and data-flow mechanics without downloading weights or datasets. The transfer-learning script uses `weights=None` by default for the same reason; replace it with a torchvision pretrained weight enum when an online checkpoint is intentionally wanted.

## Run one module

From the repository root:

```bash
uv run python codes/unit02/convolution_and_pooling.py
uv run python codes/unit02/basic_cnn.py
uv run python codes/unit02/transfer_learning.py
uv run python codes/unit02/vgg.py
uv run python codes/unit02/googlenet.py
uv run python codes/unit02/resnet.py
uv run python codes/unit02/mobilenet.py
uv run python codes/unit02/yolo.py
uv run python codes/unit02/convnext.py
```

All demos use deterministic synthetic tensors and CPU-compatible operations. No training, network access, or external files are required.
