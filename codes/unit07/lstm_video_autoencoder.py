"""Unsupervised LSTM video representation via sequence reconstruction.

Run: uv run python codes/unit07/lstm_video_autoencoder.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LSTMVideoAutoencoder(nn.Module):
    """Reconstruct frame features `[N,T,F]` through fixed representation `[N,Z]`.

    Inputs are features from a frame CNN (raw pixels are intentionally omitted).
    The encoder's final hidden state is repeated as decoder input at each step.
    Output and target both have `[N,T,F]`, enabling label-free MSE training.
    """
    def __init__(self,feature_dim:int=24,latent_dim:int=32)->None:
        super().__init__(); self.encoder=nn.LSTM(feature_dim,latent_dim,batch_first=True)
        self.decoder=nn.LSTM(latent_dim,latent_dim,batch_first=True); self.to_feature=nn.Linear(latent_dim,feature_dim)

    def forward_with_shapes(self,features:Tensor)->dict[str,Tensor]:
        encoded,(hidden,cell)=self.encoder(features); representation=hidden[-1] # [N,Z]
        decoder_input=representation[:,None,:].expand(-1,features.shape[1],-1) # [N,T,Z]
        decoded,_=self.decoder(decoder_input,(hidden,cell)); reconstruction=self.to_feature(decoded)
        return {"features":features,"encoder_sequence":encoded,"representation":representation,
                "decoder_input":decoder_input,"decoder_sequence":decoded,"reconstruction":reconstruction}
    def forward(self,features:Tensor)->Tensor:return self.forward_with_shapes(features)["reconstruction"]


if __name__ == "__main__":
    torch.manual_seed(0); features=torch.randn(2,6,24); trace=LSTMVideoAutoencoder().forward_with_shapes(features)
    for name,value in trace.items(): print(f"{name:17} shape={tuple(value.shape)}, dtype={value.dtype}")
    print(f"reconstruction MSE shape=(), value={F.mse_loss(trace['reconstruction'],features):.4f}")
