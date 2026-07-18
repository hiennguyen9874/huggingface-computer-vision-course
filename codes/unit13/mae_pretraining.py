"""Masked Autoencoder (MAE), the strong visual pretext task used by Hiera.

MAE drops most patch tokens before the encoder, inserts learned mask tokens for
the decoder, and reconstructs only missing RGB patches. This complements
`hiera.py`: MAE is a training objective, not a required inference-time module.

Run: uv run --extra cpu python codes/unit13/mae_pretraining.py
Notation: N=batch, L=patches, V=visible, M=masked, E=encoder width, Q=patch pixels.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def patchify(images: Tensor, patch_size: int) -> Tensor:
    """Convert `[N,C,H,W]` to row-major RGB patch targets `[N,L,C*P*P]`."""
    if images.ndim != 4:
        raise ValueError(f"expected [N,C,H,W], got {tuple(images.shape)}")
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("height and width must be divisible by patch_size")
    grid_h, grid_w = height // patch_size, width // patch_size
    return (images.view(batch, channels, grid_h, patch_size, grid_w, patch_size)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, grid_h * grid_w, channels * patch_size ** 2))


def deterministic_mask_indices(num_patches: int, mask_ratio: float,
                               device: torch.device) -> tuple[Tensor, Tensor]:
    """Return visible `[V]` and masked `[M]` indices for a reproducible demo."""
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be between 0 and 1")
    masked_count = int(num_patches * mask_ratio)
    # A fixed split keeps this shape demonstration reproducible without RNG.
    # Production MAE training should sample a fresh random mask for each image.
    permutation = torch.arange(num_patches, device=device)
    masked = permutation[:masked_count]
    visible = permutation[masked_count:]
    return visible, masked


class TinyMAE(nn.Module):
    """RGB `[N,3,S,S]` -> masked-patch prediction `[N,M,3*P*P]` + loss."""

    def __init__(self, image_size: int = 32, patch_size: int = 4,
                 embed_dim: int = 48, mask_ratio: float = 0.75) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size, self.patch_size, self.mask_ratio = image_size, patch_size, mask_ratio
        self.num_patches = (image_size // patch_size) ** 2
        patch_values = 3 * patch_size ** 2
        self.patch_projection = nn.Linear(patch_values, embed_dim)
        self.encoder_position = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, 4, 4 * embed_dim, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 2,
                                             enable_nested_tensor=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_position = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            embed_dim, 4, 4 * embed_dim, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, 1,
                                             enable_nested_tensor=False)
        self.pixel_head = nn.Linear(embed_dim, patch_values)

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        expected = (3, self.image_size, self.image_size)
        if images.ndim != 4 or images.shape[1:] != expected:
            raise ValueError(f"expected [N,3,{self.image_size},{self.image_size}], "
                             f"got {tuple(images.shape)}")
        patches = patchify(images, self.patch_size)                          # [N,L,Q]
        embedded = self.patch_projection(patches)                            # [N,L,E]
        visible_idx, masked_idx = deterministic_mask_indices(
            self.num_patches, self.mask_ratio, images.device
        )
        visible = embedded.index_select(1, visible_idx)
        visible = visible + self.encoder_position.index_select(1, visible_idx) # [N,V,E]
        encoded_visible = self.encoder(visible)                              # [N,V,E]

        # Restore a full row-major sequence for the lightweight decoder.
        decoder_tokens = self.mask_token.expand(images.shape[0],
                                                self.num_patches, -1).clone() # [N,L,E]
        decoder_tokens[:, visible_idx] = encoded_visible
        decoder_tokens = decoder_tokens + self.decoder_position             # [N,L,E]
        decoded = self.decoder(decoder_tokens)                               # [N,L,E]
        masked_decoded = decoded.index_select(1, masked_idx)                 # [N,M,E]
        predicted_patches = self.pixel_head(masked_decoded)                  # [N,M,Q]
        target_patches = patches.index_select(1, masked_idx)                 # [N,M,Q]
        reconstruction_loss = F.mse_loss(predicted_patches, target_patches)  # scalar
        return {"images": images, "all_pixel_patches": patches,
                "all_patch_embeddings": embedded, "visible_indices": visible_idx,
                "masked_indices": masked_idx, "visible_tokens": visible,
                "encoded_visible": encoded_visible, "decoder_tokens": decoder_tokens,
                "decoded_all": decoded, "predicted_masked_patches": predicted_patches,
                "target_masked_patches": target_patches,
                "reconstruction_loss": reconstruction_loss}

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_with_shapes(images)["reconstruction_loss"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyMAE()
    trace = model.forward_with_shapes(torch.randn(2, 3, 32, 32))
    trace["reconstruction_loss"].backward()
    for name, value in trace.items():
        print(f"{name:26} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("The encoder processes", trace["visible_tokens"].shape[1], "of",
          trace["all_pixel_patches"].shape[1], "patches (25%).")
