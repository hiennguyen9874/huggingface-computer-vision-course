"""ViViT full space-time attention and efficient factorized encoder.

Run: uv run python codes/unit07/vivit.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


def encoder(dim:int,heads:int=4)->nn.TransformerEncoder:
    layer=nn.TransformerEncoderLayer(dim,heads,2*dim,batch_first=True,dropout=0.0)
    return nn.TransformerEncoder(layer,1,enable_nested_tensor=False)


class VideoTokenizer(nn.Module):
    """Patch every frame: `[N,3,T,H,W] -> [N,T,P,D]`, P=(H/p)*(W/p)."""
    def __init__(self,dim:int=32,patch:int=8)->None:
        super().__init__(); self.patch=nn.Conv2d(3,dim,patch,patch)
    def forward(self,video:Tensor)->Tensor:
        n,c,t,h,w=video.shape; maps=self.patch(video.transpose(1,2).reshape(n*t,c,h,w)) # [N*T,D,h,w]
        return maps.flatten(2).transpose(1,2).reshape(n,t,-1,maps.shape[1])              # [N,T,P,D]


class FullAttentionViViT(nn.Module):
    """All `T*P` tokens attend jointly; input `[N,3,T,H,W]`, logits `[N,K]`."""
    def __init__(self,dim:int=32,num_classes:int=5,max_tokens:int=128)->None:
        super().__init__(); self.tokenizer=VideoTokenizer(dim); self.position=nn.Parameter(torch.randn(1,max_tokens,dim)*.02)
        self.encoder=encoder(dim); self.head=nn.Linear(dim,num_classes)
    def forward_with_shapes(self,video:Tensor)->dict[str,Tensor]:
        grid=self.tokenizer(video); tokens=grid.flatten(1,2)
        if tokens.shape[1]>self.position.shape[1]: raise ValueError("video creates more tokens than max_tokens")
        contextual=self.encoder(tokens+self.position[:,:tokens.shape[1]])
        return {"video":video,"token_grid":grid,"flat_tokens":tokens,"contextual_tokens":contextual,
                "pooled":contextual.mean(1),"logits":self.head(contextual.mean(1))}
    def forward(self,video:Tensor)->Tensor:return self.forward_with_shapes(video)["logits"]


class FactorizedViViT(nn.Module):
    """Spatial attention per frame, then temporal attention over frame summaries."""
    def __init__(self,dim:int=32,num_classes:int=5,max_patches:int=32,max_frames:int=16)->None:
        super().__init__(); self.tokenizer=VideoTokenizer(dim); self.spatial_pos=nn.Parameter(torch.randn(1,max_patches,dim)*.02)
        self.temporal_pos=nn.Parameter(torch.randn(1,max_frames,dim)*.02); self.spatial=encoder(dim); self.temporal=encoder(dim); self.head=nn.Linear(dim,num_classes)
    def forward_with_shapes(self,video:Tensor)->dict[str,Tensor]:
        grid=self.tokenizer(video); n,t,p,d=grid.shape
        spatial=self.spatial(grid.reshape(n*t,p,d)+self.spatial_pos[:,:p]) # independent frame batches
        frame_embeddings=spatial.mean(1).reshape(n,t,d)                    # one vector/frame
        temporal=self.temporal(frame_embeddings+self.temporal_pos[:,:t]); pooled=temporal.mean(1)
        return {"video":video,"token_grid":grid,"spatial_tokens":spatial,"frame_embeddings":frame_embeddings,
                "temporal_tokens":temporal,"pooled":pooled,"logits":self.head(pooled)}
    def forward(self,video:Tensor)->Tensor:return self.forward_with_shapes(video)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0); video=torch.randn(2,3,4,32,32)
    for title,model in (("full attention",FullAttentionViViT()),("factorized",FactorizedViViT())):
        model.eval()
        with torch.no_grad(): trace=model.forward_with_shapes(video)
        print(f"\n{title}")
        for name,value in trace.items(): print(f"  {name:18} shape={tuple(value.shape)}")
