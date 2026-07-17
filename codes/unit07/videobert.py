"""Tiny VideoBERT objectives: alignment, masked text, and masked visual tokens.

Run: uv run python codes/unit07/videobert.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TinyVideoBERT(nn.Module):
    """Jointly encode discrete text `[N,Lt]` and visual IDs `[N,Lv]`.

    IDs are int64. Visual IDs stand for clustered S3D features from the paper.
    Outputs token logits and one alignment logit per video-text pair.
    """
    def __init__(self,text_vocab:int=40,visual_vocab:int=24,dim:int=32)->None:
        super().__init__(); self.text_vocab=text_vocab; self.visual_vocab=visual_vocab
        self.text_embedding=nn.Embedding(text_vocab,dim); self.visual_embedding=nn.Embedding(visual_vocab,dim); self.modality=nn.Embedding(2,dim)
        layer=nn.TransformerEncoderLayer(dim,4,64,batch_first=True,dropout=0); self.encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.text_head=nn.Linear(dim,text_vocab); self.visual_head=nn.Linear(dim,visual_vocab); self.alignment_head=nn.Linear(dim,1)
    def forward(self,text_ids:Tensor,visual_ids:Tensor)->dict[str,Tensor]:
        n,lt=text_ids.shape; lv=visual_ids.shape[1]
        text=self.text_embedding(text_ids)+self.modality.weight[0]; visual=self.visual_embedding(visual_ids)+self.modality.weight[1]
        joint=self.encoder(torch.cat((text,visual),1)); pooled=joint.mean(1)
        return {"text_embeddings":text,"visual_embeddings":visual,"joint_tokens":joint,
                "text_logits":self.text_head(joint[:,:lt]),"visual_logits":self.visual_head(joint[:,lt:lt+lv]),
                "alignment_logits":self.alignment_head(pooled).squeeze(1)}


if __name__ == "__main__":
    torch.manual_seed(0); model=TinyVideoBERT(); text=torch.randint(0,40,(2,6)); visual=torch.randint(0,24,(2,4)); output=model(text,visual)
    text_targets=torch.randint(0,40,(2,6)); visual_targets=torch.randint(0,24,(2,4)); aligned=torch.tensor([1.,0.])
    losses={"masked_language":F.cross_entropy(output["text_logits"].flatten(0,1),text_targets.flatten()),
            "masked_frames":F.cross_entropy(output["visual_logits"].flatten(0,1),visual_targets.flatten()),
            "alignment":F.binary_cross_entropy_with_logits(output["alignment_logits"],aligned)}
    print(f"text IDs           shape={tuple(text.shape)}, dtype={text.dtype}")
    print(f"visual IDs         shape={tuple(visual.shape)}, dtype={visual.dtype}")
    for name,value in output.items(): print(f"{name:18} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("losses (scalars): "+", ".join(f"{k}={v:.4f}" for k,v in losses.items()))
