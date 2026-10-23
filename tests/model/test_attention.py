"""
tests/model/test_attention.py : Tests for causal SDPA attention and causal mask leak prevention.
"""

import torch

from src.rgpt.model.attention import CausalSelfAttention
from src.rgpt.model.rope import RotaryEmbedding


def test_causal_mask_leak_invariant() -> None:
    """Perturbing token at position t+1 must strictly leave attention outputs at positions <= t unchanged."""
    d_model = 128
    n_heads = 2
    n_kv_heads = 2
    head_dim = 64
    seq_len = 10

    attn = CausalSelfAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_seq_len=64,
    )
    attn.eval()
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=64)

    torch.manual_seed(42)
    x1 = torch.randn(1, seq_len, d_model)

    with torch.no_grad():
        out1 = attn(x1, rope)

    # Perturb all tokens from position 5 onwards
    x2 = x1.clone()
    x2[0, 5:] = torch.randn(seq_len - 5, d_model) * 10.0

    with torch.no_grad():
        out2 = attn(x2, rope)

    # Outputs at positions 0..4 MUST be strictly identical
    diff_past = (out1[0, :5] - out2[0, :5]).abs().max().item()
    assert diff_past < 1e-5, f"Future tokens leaked into past! Max diff: {diff_past}"

    # Outputs at position >= 5 SHOULD be different
    diff_future = (out1[0, 5:] - out2[0, 5:]).abs().max().item()
    assert diff_future > 1e-2, "Perturbed tokens produced identical output"
