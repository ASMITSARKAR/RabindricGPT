"""
src/rgpt/model/attention.py : Causal Scaled Dot-Product Attention with optional GQA and fused SDPA execution.
Backlog/Phase : I7 / Phase H      Docs : 15_BUILD_DIRECTIVE.md Phase H; 05_MODEL_SPEC.md; DR-6
Contract      : x: Tensor, rope: RotaryEmbedding -> out: Tensor
Invariants    : Causal mask leak test passes (perturb token t+1, logits <= t unchanged); SDPA fused kernel
Failure modes : Mask leakage across future tokens; dimension mismatch between query and kv projections
Status        : TESTED
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rgpt.model.rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer utilizing PyTorch's native SDPA."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int = 64,
        max_seq_len: int = 3072,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with causal SDPA."""
        batch_size, seq_len, _ = x.shape

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to query and key
        q, k = rope(q, k, seq_len=seq_len)

        # Repeat KV heads if GQA (n_kv_heads < n_heads)
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            k = torch.repeat_interleave(k, repeats=repeats, dim=1)
            v = torch.repeat_interleave(v, repeats=repeats, dim=1)

        # Fused SDPA with causal masking
        is_causal = attention_mask is None
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=is_causal
        )

        # Transpose back and project to output dimension
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_out)
