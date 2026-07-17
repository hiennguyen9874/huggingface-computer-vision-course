"""TimeSFormer-style divided space-time attention.

Run: uv run python codes/unit07/timesformer.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class DividedSpaceTimeBlock(nn.Module):
    """Attend time at fixed patch positions, then space within fixed frames.

    Input/output token grid `[N,T,P,D]`. Reshaping axes turns the same attention
    primitive into temporal batches `[N*P,T,D]` and spatial batches `[N*T,P,D]`.
    """
    def __init__(self,dim:int=32,heads:int=4)->None:
        super().__init__(); self.temporal=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.spatial=nn.MultiheadAttention(dim,heads,batch_first=True); self.norm_t=nn.LayerNorm(dim); self.norm_s=nn.LayerNorm(dim)
    def forward_with_shapes(self,x:Tensor)->dict[str,Tensor]:
        n,t,p,d=x.shape; temporal_input=x.permute(0,2,1,3).reshape(n*p,t,d)
        attended,_=self.temporal(temporal_input,temporal_input,temporal_input); temporal=self.norm_t(temporal_input+attended)
        temporal_grid=temporal.reshape(n,p,t,d).permute(0,2,1,3); spatial_input=temporal_grid.reshape(n*t,p,d)
        attended,_=self.spatial(spatial_input,spatial_input,spatial_input); spatial=self.norm_s(spatial_input+attended)
        return {"token_grid":x,"temporal_batches":temporal,"after_temporal":temporal_grid,
                "spatial_batches":spatial,"output_grid":spatial.reshape(n,t,p,d)}
    def forward(self,x:Tensor)->Tensor:return self.forward_with_shapes(x)["output_grid"]


class TinyTimeSFormer(nn.Module):
    """Patch-tokenize `[N,3,T,H,W]`, divided attention, return `[N,K]`."""
    def __init__(self,dim:int=32,num_classes:int=5)->None:
        super().__init__(); self.patch=nn.Conv2d(3,dim,8,8); self.block=DividedSpaceTimeBlock(dim); self.head=nn.Linear(dim,num_classes)
    def forward(self,video:Tensor)->Tensor:
        n,c,t,h,w=video.shape; maps=self.patch(video.transpose(1,2).reshape(n*t,c,h,w)); grid=maps.flatten(2).transpose(1,2).reshape(n,t,-1,maps.shape[1])
        return self.head(self.block(grid).mean((1,2)))


if __name__ == "__main__":
    torch.manual_seed(0); video=torch.randn(2,3,4,32,32); model=TinyTimeSFormer().eval()
    with torch.no_grad():
        n,c,t,h,w=video.shape; maps=model.patch(video.transpose(1,2).reshape(n*t,c,h,w)); grid=maps.flatten(2).transpose(1,2).reshape(n,t,-1,maps.shape[1]); trace=model.block.forward_with_shapes(grid); logits=model.head(trace["output_grid"].mean((1,2)))
    print(f"video              shape={tuple(video.shape)}")
    for name,value in trace.items(): print(f"{name:18} shape={tuple(value.shape)}")
    print(f"logits             shape={tuple(logits.shape)}")
