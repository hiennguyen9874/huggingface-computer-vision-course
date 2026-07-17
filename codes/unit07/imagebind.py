"""ImageBind concept: align many modalities through an image anchor.

Run: uv run python codes/unit07/imagebind.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TinyImageBind(nn.Module):
    """Project pooled modality features `[N,Fm]` into shared unit vectors `[N,D]`.

    Only image-other pairs need supervision. Audio and text become indirectly
    comparable because both are constrained against their corresponding images.
    """
    def __init__(self,embed_dim:int=32)->None:
        super().__init__(); self.encoders=nn.ModuleDict({"image":nn.Linear(24,embed_dim),"text":nn.Linear(20,embed_dim),"audio":nn.Linear(16,embed_dim),"depth":nn.Linear(12,embed_dim)})
    def forward(self,features:dict[str,Tensor])->dict[str,Tensor]:
        unknown=set(features)-set(self.encoders)
        if unknown: raise ValueError(f"unsupported modalities: {sorted(unknown)}")
        return {name:F.normalize(self.encoders[name](value),dim=1) for name,value in features.items()}


def info_nce(anchor:Tensor,other:Tensor,temperature:float=.07)->Tensor:
    """Symmetric image-other alignment; paired examples occupy equal row indices."""
    logits=anchor@other.T/temperature; targets=torch.arange(len(anchor),device=anchor.device)
    return (F.cross_entropy(logits,targets)+F.cross_entropy(logits.T,targets))/2


if __name__ == "__main__":
    torch.manual_seed(0); raw={"image":torch.randn(3,24),"text":torch.randn(3,20),"audio":torch.randn(3,16),"depth":torch.randn(3,12)}
    embeddings=TinyImageBind()(raw); image=embeddings["image"]
    losses={name:info_nce(image,value) for name,value in embeddings.items() if name!="image"}
    for name in raw: print(f"{name:6} raw={tuple(raw[name].shape)} -> shared={tuple(embeddings[name].shape)}, norms={embeddings[name].norm(dim=1).tolist()}")
    print("image-centered scalar losses: "+", ".join(f"{k}={v:.4f}" for k,v in losses.items()))
    print(f"indirect text-audio similarity shape={tuple((embeddings['text'] @ embeddings['audio'].T).shape)}")
