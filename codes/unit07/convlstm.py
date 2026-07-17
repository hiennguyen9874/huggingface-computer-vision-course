"""ConvLSTM preserves 2D feature maps while updating temporal memory.

Run: uv run python codes/unit07/convlstm.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class ConvLSTMCell(nn.Module):
    """One recurrent update from input `[N,C,H,W]` and state `[N,D,H,W]`."""
    def __init__(self,input_channels:int,hidden_channels:int,kernel_size:int=3)->None:
        super().__init__(); self.hidden_channels=hidden_channels
        self.gates=nn.Conv2d(input_channels+hidden_channels,4*hidden_channels,kernel_size,padding=kernel_size//2)

    def forward(self,x:Tensor,state:tuple[Tensor,Tensor])->tuple[Tensor,Tensor]:
        hidden,cell=state; forget,input_gate,output,candidate=self.gates(torch.cat((x,hidden),1)).chunk(4,1)
        cell=torch.sigmoid(forget)*cell+torch.sigmoid(input_gate)*torch.tanh(candidate)
        hidden=torch.sigmoid(output)*torch.tanh(cell)
        return hidden,cell


class ConvLSTM(nn.Module):
    """Map a sequence `[N,T,C,H,W]` to hidden maps `[N,T,D,H,W]`."""
    def __init__(self,input_channels:int=3,hidden_channels:int=16)->None:
        super().__init__(); self.cell=ConvLSTMCell(input_channels,hidden_channels); self.hidden_channels=hidden_channels

    def forward(self,sequence:Tensor)->tuple[Tensor,tuple[Tensor,Tensor]]:
        if sequence.ndim!=5: raise ValueError(f"expected [N,T,C,H,W], got {tuple(sequence.shape)}")
        n,_,_,h,w=sequence.shape; hidden=sequence.new_zeros(n,self.hidden_channels,h,w); cell=torch.zeros_like(hidden); outputs=[]
        for frame in sequence.unbind(1): hidden,cell=self.cell(frame,(hidden,cell)); outputs.append(hidden)
        return torch.stack(outputs,1),(hidden,cell)


if __name__ == "__main__":
    torch.manual_seed(0); video=torch.randn(2,5,3,16,16); output,(hidden,cell)=ConvLSTM()(video)
    print(f"input sequence shape={tuple(video.shape)}, dtype={video.dtype}")
    print(f"all hidden    shape={tuple(output.shape)}  [N,T,D,H,W]")
    print(f"final hidden  shape={tuple(hidden.shape)}")
    print(f"final cell    shape={tuple(cell.shape)}")
