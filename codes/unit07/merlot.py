"""MERLOT-inspired temporal ordering and frame-caption matching objectives.

Run: uv run python codes/unit07/merlot.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TinyMERLOT(nn.Module):
    """Align ordered frame/text features `[N,T,F]` in shared `[N,T,D]` space.

    The ordering head predicts each frame's original temporal position `[0,T)`;
    matching uses all-pairs clip similarities `[N,N]` with diagonal positives.
    """
    def __init__(self,input_dim:int=24,embed_dim:int=32,max_frames:int=8)->None:
        super().__init__(); self.visual=nn.Linear(input_dim,embed_dim); self.text=nn.Linear(input_dim,embed_dim); self.position_head=nn.Linear(embed_dim,max_frames)
    def forward(self,frames:Tensor,captions:Tensor)->dict[str,Tensor]:
        visual=F.normalize(self.visual(frames),dim=-1); text=F.normalize(self.text(captions),dim=-1)
        similarities=visual.mean(1)@text.mean(1).T; position_logits=self.position_head(visual)[:,:,:frames.shape[1]]
        return {"visual_tokens":visual,"text_tokens":text,"matching_logits":similarities,"position_logits":position_logits}


if __name__ == "__main__":
    torch.manual_seed(0); frames=torch.randn(3,5,24); captions=frames+.1*torch.randn_like(frames); output=TinyMERLOT()(frames,captions)
    targets=torch.arange(3); positions=torch.arange(5).expand(3,5)
    matching=F.cross_entropy(output["matching_logits"],targets); ordering=F.cross_entropy(output["position_logits"].flatten(0,1),positions.flatten())
    print(f"frame features   shape={tuple(frames.shape)}; caption features={tuple(captions.shape)}")
    for name,value in output.items(): print(f"{name:16} shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"matching loss={matching:.4f}; temporal ordering loss={ordering:.4f}")
