# Unit 13 — Architecture alternatives beyond standard Transformers

Executable lessons designed from `chapters-vi/unit13.md` and
`chapters-en/unit13/`. Each file owns one model or mechanism, documents tensor
contracts in code, and prints the shape and dtype at every meaningful step.

These are deliberately small, CPU-friendly teaching implementations. They show
the defining computation and module boundaries, but they are not exact,
optimized reproductions of the research repositories or pretrained models.

| File | Model/module | Main executable contract |
|---|---|---|
| `hyena.py` | implicit filter, FFT long convolution, gating, vision model | image `[N,3,H,W]` -> patch tokens `[N,L,D]` -> logits `[N,K]` |
| `ijepa.py` | context encoder, EMA target encoder, predictor | context `[N,C,E]` -> predicted target embeddings `[N,T,E]` -> scalar loss |
| `retention.py` | multi-scale retention | tokens `[N,L,D]` -> parallel/recurrent output `[N,L,D]` |
| `rmt.py` | MaSA, decomposed MaSA, tiny RMT | token grid `[N,H,W,D]` -> spatially decayed grid -> logits `[N,K]` |
| `vir.py` | ViR-style classifier | positioned patches + final class token -> retention -> logits `[N,K]` |
| `hiera.py` | local mask units and hierarchical stages | image -> `[8x8,32]` -> `[4x4,64]` -> `[2x2,128]` -> logits |
| `mae_pretraining.py` | MAE pretraining objective used by Hiera | 25% visible patches -> reconstruct the masked 75% |

## Why these boundaries?

- Hyena's filter, FFT convolution, and gates belong together because they form
  one operator and are easiest to understand in one trace.
- Core retention is separate so `vir.py` can demonstrate that replacing
  attention is a reusable sequence operation. `rmt.py` remains separate because
  MaSA is a 2-D spatial-attention adaptation, not the same operation.
- MAE is separate from Hiera because MAE is a pretraining objective; it isn't an
  inference-time Hiera layer.
- I-JEPA keeps its three encoders/predictor in one file because their shared
  masking and stop-gradient contract defines one training step.

## Run

From the repository root, run one lesson:

```bash
uv run --extra cpu python codes/unit13/hyena.py
```

Run every lesson:

```bash
for file in codes/unit13/*.py; do
  uv run --extra cpu python "$file"
done
```

All demos use synthetic tensors and fixed random seeds, require no downloads,
and run on CPU. Notation: `N` batch, `C` channels/context count (documented per
file), `H/W` spatial dimensions, `L` token count, `D/E` feature width, `M`
heads or masked count (documented per file), and `K` classes.
