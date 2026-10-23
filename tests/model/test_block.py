"""
tests/model/test_block.py : Tests for pre-norm Transformer block and residual connections.
"""

import torch
import torch.nn as nn

from src.rgpt.model.block import TransformerBlock
from src.rgpt.model.rope import RotaryEmbedding


def test_transformer_block_residual_identity() -> None:
    """Zero-initialized weights must produce an exact identity residual mapping."""
    d_model = 128
    block = TransformerBlock(
        d_model=d_model,
        n_heads=2,
        n_kv_heads=2,
        hidden_dim=384,
        head_dim=64,
        rms_norm_eps=1e-5,
        max_seq_len=64,
    )
    rope = RotaryEmbedding(dim=64, max_seq_len=64)

    # Zero-initialize the residual output projections
    nn.init.zeros_(block.attn.out_proj.weight)
    nn.init.zeros_(block.mlp.w_down.weight)

    x = torch.randn(2, 8, d_model)
    out = block(x, rope)

    # Output must strictly equal input x
    assert torch.allclose(out, x, atol=1e-6)
