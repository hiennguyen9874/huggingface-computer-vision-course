# Unit 4 code — Multimodal text and vision

These modules turn `chapters-vi/unit04.md` and `chapters-en/unit4/` into small, executable PyTorch lessons. Every file owns one cohesive chapter concept, documents tensor types and contracts, and includes a deterministic `__main__` demo that prints each meaningful intermediate shape.

The implementations are intentionally compact architectural models, **not pretrained production checkpoints**. They need no model downloads, datasets, network access, or GPU. This makes the data flow inspectable and keeps all examples runnable offline.

## Design map

| File | Chapter section/model | What the code demonstrates |
| --- | --- | --- |
| `multimodal_fusion.py` | multimodal architecture and VLM strategies | projected vision/text tokens; early, late, and hybrid fusion |
| `contrastive_learning.py` | contrastive objectives and losses | classic pairwise margin loss and CLIP symmetric cross-entropy |
| `clip.py` | CLIP | independent image/text encoders, normalized shared space, all-pairs similarity, zero-shot probabilities |
| `image_text_retrieval.py` | image-text retrieval and multimodal search | exact cosine-similarity index for both retrieval directions |
| `vqa.py` | VQA and visual reasoning | image patches, question encoding, cross-attention, fixed-answer classification |
| `document_vqa.py` | LayoutLM, Donut, and Nougat | LayoutLM-style OCR token + bounding-box embeddings and answer-span prediction; notes the OCR-free alternative |
| `image_captioning.py` | Show-and-Tell, ViT-GPT2, GIT | visual encoder, autoregressive decoder, teacher-forced logits, greedy generation |
| `blip.py` | BLIP and CapFilt | contrastive, image-text matching, and language-generation heads plus caption filtering |
| `owl_vit.py` | visual grounding and OWL-ViT | patch-level text-query scores, objectness, normalized boxes, pixel-box conversion |
| `text_to_image.py` | autoregressive and Stable Diffusion generation | image-token prediction and text/time-conditioned latent noise prediction |
| `transfer_learning.py` | zero/few-shot and multimodal transfer learning | zero-shot inference, frozen linear probe, selective unfreezing, per-group learning rates |

Some named production models share an architecture and therefore one lesson: ViT-GPT2/GIT use `image_captioning.py`; Donut/Nougat follow the visual-encoder/autoregressive-decoder path described there; BLIP-VQA is represented by the grounded BLIP path plus `vqa.py`; visual grounding is the task implemented by `owl_vit.py`. This avoids duplicating nearly identical educational code while retaining each distinct data flow.

## Run

From the repository root:

```bash
uv run python codes/unit04/multimodal_fusion.py
uv run python codes/unit04/contrastive_learning.py
uv run python codes/unit04/clip.py
uv run python codes/unit04/image_text_retrieval.py
uv run python codes/unit04/vqa.py
uv run python codes/unit04/document_vqa.py
uv run python codes/unit04/image_captioning.py
uv run python codes/unit04/blip.py
uv run python codes/unit04/owl_vit.py
uv run python codes/unit04/text_to_image.py
uv run python codes/unit04/transfer_learning.py
```

## Shape notation

- `N`: batch size
- `C`: image/latent channels
- `H`, `W`: height and width
- `Pi`: number of image patches
- `Lt`, `Lq`, `Ld`: text, question, and document sequence lengths
- `D`: shared hidden/embedding dimension
- `Q`: number of open-vocabulary text queries
- `V`: vocabulary size

All input images in demos are synthetic tensors. Token IDs stand in for the output of a real tokenizer, and query embeddings stand in for a pretrained text encoder where a lesson focuses on the downstream module.
