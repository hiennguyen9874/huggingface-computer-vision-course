"""U-Net semantic segmentation with encoder/decoder skip connections.

Run: uv run python codes/unit06/unet.py
Input is float `[N,3,H,W]`; output is per-pixel class logits `[N,K,H,W]`.
Height and width must be divisible by 4 in this compact two-level U-Net.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Two shape-preserving 3x3 convolutions with ReLU activations."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1), nn.ReLU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class UNet(nn.Module):
    """A two-level U-Net that preserves input spatial resolution."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.encoder_1 = ConvBlock(3, 32)       # [N,3,H,W] -> [N,32,H,W]
        self.encoder_2 = ConvBlock(32, 64)      # [N,32,H/2,W/2] -> [N,64,H/2,W/2]
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 128)    # [N,64,H/4,W/4] -> [N,128,H/4,W/4]
        self.up_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_2 = ConvBlock(128, 64)     # concat(up_2, skip_2): 64+64 channels
        self.up_1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder_1 = ConvBlock(64, 32)      # concat(up_1, skip_1): 32+32 channels
        self.class_head = nn.Conv2d(32, num_classes, 1)  # pixel-wise linear classifier

    def forward_with_shapes(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images [N, 3, H, W], got {tuple(images.shape)}")
        if images.shape[-2] % 4 or images.shape[-1] % 4:
            raise ValueError("image height and width must be divisible by 4")
        skip_1 = self.encoder_1(images)                         # [N,32,H,W]
        pooled_1 = self.pool(skip_1)                            # [N,32,H/2,W/2]
        skip_2 = self.encoder_2(pooled_1)                       # [N,64,H/2,W/2]
        pooled_2 = self.pool(skip_2)                            # [N,64,H/4,W/4]
        bottleneck = self.bottleneck(pooled_2)                  # [N,128,H/4,W/4]
        up_2 = self.up_2(bottleneck)                            # [N,64,H/2,W/2]
        merged_2 = torch.cat((up_2, skip_2), dim=1)             # [N,128,H/2,W/2]
        decoded_2 = self.decoder_2(merged_2)                    # [N,64,H/2,W/2]
        up_1 = self.up_1(decoded_2)                             # [N,32,H,W]
        merged_1 = torch.cat((up_1, skip_1), dim=1)             # [N,64,H,W]
        decoded_1 = self.decoder_1(merged_1)                    # [N,32,H,W]
        logits = self.class_head(decoded_1)                     # [N,K,H,W]
        predicted_classes = logits.argmax(dim=1)                # [N,H,W], long
        return {
            "images": images, "encoder_skip_1": skip_1, "encoder_skip_2": skip_2,
            "bottleneck": bottleneck, "decoder_level_2": decoded_2,
            "decoder_level_1": decoded_1, "class_logits": logits,
            "predicted_classes": predicted_classes,
        }

    def forward(self, images: Tensor) -> Tensor:
        """Return `[N,K,H,W]` logits suitable for pixel-wise cross entropy."""
        return self.forward_with_shapes(images)["class_logits"]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = UNet(num_classes=5).eval()  # background, person, car, road, sky
    images = torch.rand(2, 3, 64, 80)
    with torch.no_grad():
        steps = model.forward_with_shapes(images)
    for name, value in steps.items():
        print(f"{name:20} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("Each output pixel has 5 logits; argmax selects its semantic class ID.")
