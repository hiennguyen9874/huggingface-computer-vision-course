"""Momentum Contrast (MoCo): momentum encoder, negative queue, InfoNCE.

Run: uv run python codes/unit07/moco.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ClipEncoder(nn.Module):
    """Encode float clips `[N,3,T,H,W]` as unit vectors `[N,D]`."""
    def __init__(self, dim: int) -> None:
        super().__init__(); self.net=nn.Sequential(nn.Conv3d(3,16,3,2,1),nn.ReLU(),nn.Conv3d(16,dim,3,2,1))
    def forward(self,x:Tensor)->Tensor:return F.normalize(self.net(x).mean((2,3,4)),dim=1)


class MoCo(nn.Module):
    """Contrast two augmented views `[N,3,T,H,W]` against a queue.

    Output logits `[N,1+Q]`: column zero is the positive key and the remaining
    columns are queued negatives. Targets `[N]` are therefore all zero.
    """
    def __init__(self, dim: int = 32, queue_size: int = 16, momentum: float = .99, temperature: float = .07) -> None:
        super().__init__(); self.query_encoder=ClipEncoder(dim); self.key_encoder=ClipEncoder(dim)
        self.key_encoder.load_state_dict(self.query_encoder.state_dict())
        for parameter in self.key_encoder.parameters(): parameter.requires_grad=False
        self.register_buffer("queue",F.normalize(torch.randn(queue_size,dim),dim=1)); self.register_buffer("pointer",torch.zeros((),dtype=torch.long))
        self.momentum=momentum; self.temperature=temperature

    @torch.no_grad()
    def update_key_encoder(self)->None:
        for query,key in zip(self.query_encoder.parameters(),self.key_encoder.parameters()):
            key.data.mul_(self.momentum).add_(query.data,alpha=1-self.momentum)

    @torch.no_grad()
    def enqueue(self,keys:Tensor)->None:
        # Circular insertion supports any batch up to queue size.
        for key in keys:
            index=int(self.pointer.item()); self.queue[index].copy_(key); self.pointer.fill_((index+1)%len(self.queue))

    def forward(self,query_view:Tensor,key_view:Tensor)->tuple[Tensor,Tensor,dict[str,Tensor]]:
        query=self.query_encoder(query_view)                        # [N,D], gradients
        with torch.no_grad(): self.update_key_encoder(); key=self.key_encoder(key_view) # [N,D]
        positive=(query*key).sum(1,keepdim=True)                    # [N,1]
        negatives=query@self.queue.detach().T                       # [N,Q]
        logits=torch.cat((positive,negatives),1)/self.temperature   # [N,1+Q]
        targets=torch.zeros(query.shape[0],dtype=torch.long,device=query.device)
        self.enqueue(key); return logits,targets,{"query":query,"key":key,"positive":positive,"negatives":negatives}


if __name__ == "__main__":
    torch.manual_seed(0); base=torch.randn(2,3,8,32,32); model=MoCo(); logits,targets,trace=model(base+.01*torch.randn_like(base),base+.01*torch.randn_like(base))
    for name,value in trace.items(): print(f"{name:10} shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"logits     shape={tuple(logits.shape)}; targets={targets.tolist()}; InfoNCE={F.cross_entropy(logits,targets):.4f}")
    print(f"queue      shape={tuple(model.queue.shape)}; next insertion index={model.pointer.item()}")
