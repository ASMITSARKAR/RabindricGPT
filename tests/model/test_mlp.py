"""
tests/model/test_mlp.py : Tests for bias-free SwiGLU feed-forward network.
"""

import torch

from src.rgpt.model.mlp import SwiGLU


def test_swiglu_no_biases() -> None:
    """SwiGLU linear projections must have bias=False."""
    mlp = SwiGLU(d_model=256, hidden_dim=704)
    assert mlp.w_gate.bias is None
    assert mlp.w_up.bias is None
    assert mlp.w_down.bias is None


def test_swiglu_shape_transformation() -> None:
    """Output tensor shape must match input tensor shape (batch, seq, d_model)."""
    mlp = SwiGLU(d_model=128, hidden_dim=384)
    x = torch.randn(2, 16, 128)
    out = mlp(x)
    assert out.shape == (2, 16, 128)
