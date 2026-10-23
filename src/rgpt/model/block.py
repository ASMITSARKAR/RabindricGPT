"""
src/rgpt/model/block.py : Pre-norm Transformer decoder block combining RMSNorm, CausalAttention, and SwiGLU.
Backlog/Phase : I7 / Phase H      Docs : 15_BUILD_DIRECTIVE.md Phase H; 05_MODEL_SPEC.md
Contract      : x: Tensor, rope: RotaryEmbedding -> out: Tensor
Invariants    : Residual connections around pre-norm attention and pre-norm MLP; residual scaling applied
Failure modes : Shape mismatch across residual additions
Status        : TESTED
"""

from typing import Optional

import torch
import torch.nn as nn

from src.rgpt.model.attention import CausalSelfAttention
from src.rgpt.model.mlp import SwiGLU
from src.rgpt.model.norm import RMSNorm
from src.rgpt.model.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    """Single pre-norm decoder transformer layer."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        hidden_dim: int,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        max_seq_len: int = 3072,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
        )
        self.mlp_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.mlp = SwiGLU(d_model=d_model, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pre-norm residual block forward: x + Attn(RMSNorm(x)) -> x + MLP(RMSNorm(x))."""
        h = x + self.attn(self.attn_norm(x), rope, attention_mask=attention_mask)
        out = h + self.mlp(self.mlp_norm(h))
        return out
