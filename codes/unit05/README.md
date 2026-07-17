# Unit 5 — Generative vision models

Small, executable PyTorch implementations matching `chapters-vi/unit05.md` and
`chapters-en/unit5/`. They teach tensor flow and objectives; they are not
pretrained image generators.

| File | Concept | Main contract |
|---|---|---|
| `generative_metrics.py` | PSNR, SSIM, FID, IS, CLIP score | images/features -> scalar metric |
| `gan.py` | DCGAN generator, discriminator, adversarial losses | noise -> image -> real/fake logit |
| `vae.py` | convolutional VAE and reparameterization | image -> `(mu, logvar)` -> reconstruction |
| `stylegan.py` | mapping network, AdaIN, noise injection | `z -> w -> styled image` |
| `diffusion.py` | DDPM forward noise and reverse sampling | `(x_t, t) -> predicted noise` |
| `stable_diffusion.py` | latent VAE, text conditioning, cross-attention denoiser | image/text -> conditioned latent noise |
| `dreambooth.py` | subject tuning with prior preservation | instance/prior batches -> denoising loss |
| `lora.py` | frozen linear layer with low-rank update | `x -> Wx + scale*BAx` |
| `controlnet.py` | zero-initialized condition residuals | noisy image + condition -> noise prediction |
| `cyclegan.py` | ResNet generators, PatchGANs, CycleGAN losses | unpaired domains X/Y -> translations/cycles |

DreamBooth is a fine-tuning **procedure**, not a new backbone architecture;
`dreambooth.py` therefore focuses on its instance and prior-preservation losses.
Parameter-efficient personalization is demonstrated separately in `lora.py`.
Ethics/privacy/bias is a deployment concern and intentionally has no model demo.

Run one lesson:

```bash
uv run python codes/unit05/gan.py
```

Run all lessons:

```bash
for file in codes/unit05/*.py; do uv run python "$file"; done
```

Notation used throughout: `N` batch, `C` channels, `H/W` image size, `D` feature
width, `L` token length, and `Z` latent width.
