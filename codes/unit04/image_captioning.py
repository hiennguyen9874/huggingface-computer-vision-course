"""Image encoder plus autoregressive text decoder for captioning.

Run with:
    uv run python codes/unit04/image_captioning.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EducationalImageCaptioner(nn.Module):
    """A small Show-and-Tell/ViT-GPT2-style image captioner.

    Inputs during training:
        images: float RGB tensor `[batch, 3, 32, 32]`.
        input_token_ids: integer caption prefix `[batch, caption_length]`.
    Output:
        next-token logits `[batch, caption_length, vocabulary_size]`.

    The visual vector initializes the GRU hidden state. At generation time the
    model feeds each predicted token back as the next input.
    """

    def __init__(self, vocabulary_size: int = 50, hidden_dim: int = 32) -> None:
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=4),  # [N, 3, 32, 32] -> [N, 32, 8, 8]
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # -> [N, 32, 1, 1]
            nn.Flatten(),  # -> [N, 32]
            nn.Linear(32, hidden_dim),
        )
        self.token_embedding = nn.Embedding(vocabulary_size, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.vocabulary_head = nn.Linear(hidden_dim, vocabulary_size)

    def forward_with_shapes(self, images: Tensor, input_token_ids: Tensor) -> dict[str, Tensor]:
        visual_context = self.visual_encoder(images)  # [N, 3, H, W] -> [N, D]
        token_embeddings = self.token_embedding(input_token_ids)  # [N, L] -> [N, L, D]
        hidden_state = visual_context.unsqueeze(0)  # GRU state [1, N, D]
        decoder_states, _ = self.decoder(token_embeddings, hidden_state)  # [N, L, D]
        token_logits = self.vocabulary_head(decoder_states)  # [N, L, V]
        return {
            "images": images,
            "input_token_ids": input_token_ids,
            "visual_context": visual_context,
            "token_embeddings": token_embeddings,
            "decoder_states": decoder_states,
            "token_logits": token_logits,
        }

    def forward(self, images: Tensor, input_token_ids: Tensor) -> Tensor:
        return self.forward_with_shapes(images, input_token_ids)["token_logits"]

    @torch.no_grad()
    def generate(self, images: Tensor, bos_token_id: int, max_new_tokens: int = 8) -> Tensor:
        """Greedily generate token IDs `[batch, max_new_tokens]`."""
        hidden = self.visual_encoder(images).unsqueeze(0)  # [1, N, D]
        current = torch.full(
            (images.shape[0], 1), bos_token_id, dtype=torch.long, device=images.device
        )  # [N, 1]
        generated: list[Tensor] = []
        for _ in range(max_new_tokens):
            embedded = self.token_embedding(current[:, -1:])  # [N, 1, D]
            state, hidden = self.decoder(embedded, hidden)  # state [N, 1, D]
            next_token = self.vocabulary_head(state[:, -1]).argmax(dim=-1)  # [N]
            generated.append(next_token)
            current = torch.cat((current, next_token[:, None]), dim=1)
        return torch.stack(generated, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = EducationalImageCaptioner().eval()
    images = torch.randn(2, 3, 32, 32)
    caption_prefix = torch.randint(0, 50, (2, 6), dtype=torch.long)
    with torch.no_grad():
        steps = model.forward_with_shapes(images, caption_prefix)
        generated = model.generate(images, bos_token_id=1, max_new_tokens=8)
    for name, tensor in steps.items():
        print(f"{name:20} shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"generated_token_ids  shape={tuple(generated.shape)}, dtype={generated.dtype}")
    print(generated)
