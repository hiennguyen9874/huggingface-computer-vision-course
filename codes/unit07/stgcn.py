"""Spatial-Temporal Graph CNN for skeleton action classification.

Run: uv run python codes/unit07/stgcn.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class STGCNBlock(nn.Module):
    """Propagate across joints then convolve time.

    Input/output `[N,C,T,V]`; adjacency `[V,V]` maps neighboring body joints.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(); self.channel_mix=nn.Conv2d(in_channels,out_channels,1)
        self.temporal=nn.Conv2d(out_channels,out_channels,(3,1),padding=(1,0)); self.activation=nn.ReLU()

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        spatial = torch.einsum("nctv,vw->nctw", x, adjacency) # aggregate graph neighbors
        return self.activation(self.temporal(self.channel_mix(spatial)))


class TinySTGCN(nn.Module):
    """Classify skeleton coordinates `[N,2,T,V]` into logits `[N,K]`."""
    def __init__(self, adjacency: Tensor, num_classes: int = 5) -> None:
        super().__init__(); degree=adjacency.sum(1).clamp_min(1)
        self.register_buffer("adjacency", adjacency / degree[:,None])
        self.block1=STGCNBlock(2,16); self.block2=STGCNBlock(16,32); self.head=nn.Linear(32,num_classes)

    def forward_with_shapes(self,x:Tensor)->dict[str,Tensor]:
        first=self.block1(x,self.adjacency); second=self.block2(first,self.adjacency); pooled=second.mean((2,3))
        return {"joints":x,"adjacency":self.adjacency,"stgcn_1":first,"stgcn_2":second,
                "pooled":pooled,"logits":self.head(pooled)}
    def forward(self,x:Tensor)->Tensor:return self.forward_with_shapes(x)["logits"]


if __name__ == "__main__":
    # Five-joint chain: head--shoulder--hip--knee--ankle, with self edges.
    adjacency=torch.eye(5)
    for i in range(4): adjacency[i,i+1]=adjacency[i+1,i]=1
    torch.manual_seed(0); trace=TinySTGCN(adjacency).forward_with_shapes(torch.randn(2,2,8,5))
    for name,value in trace.items(): print(f"{name:10} shape={tuple(value.shape)}, dtype={value.dtype}")
