# Unit 7 — Video processing

Small, executable PyTorch implementations matching `chapters-vi/unit07.md` and
`chapters-en/unit7/video-processing/`. They preserve each architecture's core
data flow, but are teaching models rather than pretrained paper reproductions.

## Design map

| File | Lesson / model | Main tensor contract |
|---|---|---|
| `video_sampling.py` | video layout and uniform frame sampling | `[T,H,W,C] -> [S,H,W,C]` |
| `tubelet_embedding.py` | 3D tubelet tokenization | `[N,C,T,H,W] -> [N,L,D]` |
| `two_stream_network.py` | RGB + motion streams | two `[N,C,T,H,W]` clips -> logits |
| `resnet3d.py` | residual 3D CNN | `[N,3,T,H,W] -> [N,K]` |
| `r2plus1d.py` | factorized spatial/temporal convolution | `[N,C,T,H,W] -> [N,K]` |
| `moco.py` | momentum encoder, queue, InfoNCE | two clip views -> logits/targets |
| `x3d.py` | lightweight depthwise 3D CNN | `[N,3,T,H,W] -> [N,K]` |
| `stgcn.py` | skeleton spatial-temporal GCN | `[N,C,T,V] -> [N,K]` |
| `lrcn.py` | frame CNN + LSTM | `[N,T,C,H,W] -> [N,K]` |
| `convlstm.py` | convolutional LSTM cell | `[N,T,C,H,W] -> [N,T,D,H,W]` |
| `lstm_video_autoencoder.py` | unsupervised sequence reconstruction | `[N,T,F] -> reconstruction` |
| `rnn_temporal_attention.py` | temporal attention over frame features | `[N,T,F] -> context/logits` |
| `vivit.py` | full and factorized ViViT | `[N,C,T,H,W] -> [N,K]` |
| `timesformer.py` | divided temporal then spatial attention | `[N,C,T,H,W] -> [N,K]` |
| `videobert.py` | joint video-text masked modeling/alignment | token IDs -> three objectives |
| `merlot.py` | frame-caption matching and temporal ordering | frame/text features -> losses |
| `vatt.py` | visual-audio-text contrastive alignment | three modalities -> embeddings/loss |
| `video_llama.py` | vision/audio query-to-language adapters | modalities -> language tokens |
| `imagebind.py` | image-centered multimodal InfoNCE | modality features -> shared space |

Notation: `N` batch, `T` time, `C` channels, `H/W` spatial size, `V` graph
vertices, `L` tokens, `D` embedding width, `F` features, and `K` classes.
All inputs are floating point except explicitly documented token/class indices.

Run one lesson:

```bash
uv run python codes/unit07/resnet3d.py
```

Run all lessons:

```bash
for file in codes/unit07/*.py; do uv run python "$file"; done
```
