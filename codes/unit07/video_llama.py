"""Video-LLaMA concept: visual/audio branches produce language-space tokens.

Run: uv run python codes/unit07/video_llama.py
"""
from __future__ import annotations
import torch
from torch import Tensor, nn


class QueryAdapter(nn.Module):
    """Compress modality tokens `[N,L,Dm]` into learned queries `[N,Q,Dlang]`.

    Cross-attention lets each learned query select useful modality evidence. The
    projection is the alignment bridge to a frozen LLM's embedding width.
    """
    def __init__(self,modality_dim:int,hidden_dim:int=32,language_dim:int=48,num_queries:int=4)->None:
        super().__init__(); self.input_projection=nn.Linear(modality_dim,hidden_dim); self.queries=nn.Parameter(torch.randn(1,num_queries,hidden_dim)*.02)
        self.cross_attention=nn.MultiheadAttention(hidden_dim,4,batch_first=True); self.to_language=nn.Linear(hidden_dim,language_dim)
    def forward(self,tokens:Tensor)->tuple[Tensor,Tensor]:
        memory=self.input_projection(tokens); queries=self.queries.expand(tokens.shape[0],-1,-1)
        attended,weights=self.cross_attention(queries,memory,memory)
        return self.to_language(attended),weights


class TinyVideoLLaMA(nn.Module):
    """Create visual and audio prefix tokens consumed by an LLM (LLM omitted)."""
    def __init__(self)->None:
        super().__init__(); self.vision_branch=QueryAdapter(24); self.audio_branch=QueryAdapter(16)
    def forward(self,video_tokens:Tensor,audio_tokens:Tensor)->dict[str,Tensor]:
        visual,v_weights=self.vision_branch(video_tokens); audio,a_weights=self.audio_branch(audio_tokens)
        return {"visual_language_tokens":visual,"audio_language_tokens":audio,
                "llm_prefix":torch.cat((visual,audio),1),"visual_attention":v_weights,"audio_attention":a_weights}


if __name__ == "__main__":
    torch.manual_seed(0); video=torch.randn(2,8,24); audio=torch.randn(2,12,16); output=TinyVideoLLaMA()(video,audio)
    print(f"video tokens shape={tuple(video.shape)}; audio tokens shape={tuple(audio.shape)}")
    for name,value in output.items(): print(f"{name:22} shape={tuple(value.shape)}, dtype={value.dtype}")
    print("The [N,8,48] prefix would be prepended to text embeddings before the frozen LLM.")
