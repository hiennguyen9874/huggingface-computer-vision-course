"""Long-term Recurrent Convolutional Network (LRCN): CNN then LSTM.

Run: uv run python codes/unit07/lrcn.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class LRCN(nn.Module):
    """Classify frame-major videos `[N,T,3,H,W]` into logits `[N,K]`.

    The same 2D CNN extracts a `D`-vector from every frame. LSTM consumes the
    resulting `[N,T,D]` sequence; its final output summarizes clip history.
    """
    def __init__(self, feature_dim: int = 32, hidden_dim: int = 48, num_classes: int = 5) -> None:
        super().__init__(); self.cnn=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,feature_dim,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.lstm=nn.LSTM(feature_dim,hidden_dim,batch_first=True); self.classifier=nn.Linear(hidden_dim,num_classes)

    def forward_with_shapes(self,video:Tensor)->dict[str,Tensor]:
        if video.ndim!=5 or video.shape[2]!=3: raise ValueError(f"expected [N,T,3,H,W], got {tuple(video.shape)}")
        n,t,c,h,w=video.shape; maps=self.cnn(video.reshape(n*t,c,h,w)) # [N*T,D,1,1]
        frame_features=maps.flatten(1).reshape(n,t,-1)                # [N,T,D]
        sequence,(hidden,_)=self.lstm(frame_features)                 # [N,T,R], [1,N,R]
        logits=self.classifier(sequence[:,-1])
        return {"video":video,"cnn_maps":maps,"frame_features":frame_features,"lstm_sequence":sequence,"final_hidden":hidden,"logits":logits}
    def forward(self,video:Tensor)->Tensor:return self.forward_with_shapes(video)["logits"]


if __name__ == "__main__":
    torch.manual_seed(0); model=LRCN().eval()
    with torch.no_grad(): trace=model.forward_with_shapes(torch.randn(2,6,3,32,32))
    for name,value in trace.items(): print(f"{name:15} shape={tuple(value.shape)}, dtype={value.dtype}")
