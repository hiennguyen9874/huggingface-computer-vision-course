"""VATT-inspired visual-audio-text encoders, Droptoken, contrastive loss.

Run: uv run python codes/unit07/vatt.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ModalityEncoder(nn.Module):
    """Project modality tokens `[N,L,F]`, encode, and pool to unit `[N,D]`."""
    def __init__(self,input_dim:int,dim:int=32)->None:
        super().__init__(); self.projection=nn.Linear(input_dim,dim)
        layer=nn.TransformerEncoderLayer(dim,4,64,batch_first=True,dropout=0); self.encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
    def forward(self,tokens:Tensor,keep_every:int=2)->tuple[Tensor,Tensor]:
        kept=tokens[:,::keep_every]                    # Droptoken: deterministic subset [N,L',F]
        encoded=self.encoder(self.projection(kept))    # [N,L',D]
        return F.normalize(encoded.mean(1),dim=1),kept


class TinyVATT(nn.Module):
    """Map video/audio/text token features to a common embedding space."""
    def __init__(self)->None:
        super().__init__(); self.video=ModalityEncoder(24); self.audio=ModalityEncoder(16); self.text=ModalityEncoder(20)
    def forward(self,video:Tensor,audio:Tensor,text:Tensor)->dict[str,Tensor]:
        v,vk=self.video(video); a,ak=self.audio(audio); t,tk=self.text(text)
        return {"video_kept":vk,"audio_kept":ak,"text_kept":tk,"video_embedding":v,"audio_embedding":a,"text_embedding":t}


def symmetric_nce(first:Tensor,second:Tensor,temperature:float=.1)->Tensor:
    """InfoNCE scalar for aligned rows of two `[N,D]` modality embeddings."""
    logits=first@second.T/temperature; targets=torch.arange(len(first),device=first.device)
    return (F.cross_entropy(logits,targets)+F.cross_entropy(logits.T,targets))/2


if __name__ == "__main__":
    torch.manual_seed(0); output=TinyVATT()(torch.randn(3,8,24),torch.randn(3,12,16),torch.randn(3,6,20))
    for name,value in output.items(): print(f"{name:16} shape={tuple(value.shape)}, dtype={value.dtype}")
    va=symmetric_nce(output["video_embedding"],output["audio_embedding"]); vt=symmetric_nce(output["video_embedding"],output["text_embedding"])
    print(f"video-audio NCE={va:.4f}; video-text NCE={vt:.4f}; total={va+vt:.4f}")
