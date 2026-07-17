"""Temporal attention selects important frame features after an RNN.

Run: uv run python codes/unit07/rnn_temporal_attention.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class TemporalAttentionRNN(nn.Module):
    """Map frame features `[N,T,F]` to context `[N,D]` and logits `[N,K]`.

    Softmax weights `[N,T]` sum to one for each clip, making the selected frames
    inspectable. Spatial attention would analogously retain an H/W axis.
    """
    def __init__(self,feature_dim:int=24,hidden_dim:int=32,num_classes:int=5)->None:
        super().__init__(); self.rnn=nn.GRU(feature_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.score=nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1)); self.head=nn.Linear(2*hidden_dim,num_classes)

    def forward(self,features:Tensor)->tuple[Tensor,Tensor,Tensor]:
        sequence,_=self.rnn(features)                    # [N,T,2D]
        weights=self.score(sequence).squeeze(-1).softmax(1) # [N,T]
        context=torch.sum(sequence*weights[:,:,None],dim=1) # [N,2D]
        return self.head(context),weights,context


if __name__ == "__main__":
    torch.manual_seed(0); features=torch.randn(2,6,24); logits,weights,context=TemporalAttentionRNN()(features)
    print(f"frame features shape={tuple(features.shape)}, dtype={features.dtype}")
    print(f"attention      shape={tuple(weights.shape)}, sums={weights.sum(1).tolist()}")
    print(f"context        shape={tuple(context.shape)}")
    print(f"class logits   shape={tuple(logits.shape)}")
