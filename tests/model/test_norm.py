"""
tests/model/test_norm.py : Tests for RMSNorm numerical stability and invariance to global scale.
"""

import torch

from src.rgpt.model.norm import RMSNorm


def test_rmsnorm_output_scale_invariance() -> None:
    """RMSNorm must normalize input activations to unit root-mean-square variance."""
    dim = 64
    norm = RMSNorm(dim=dim, eps=1e-5)
    x = torch.randn(2, 10, dim) * 5.0 + 3.0

    out = norm(x)
    assert out.shape == x.shape

    # RMS of normalized output (with unit weight) should be approx 1.0
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)

    # Scaling input x by scalar c leaves direction normalized
    c = 10.0
    out_scaled = norm(x * c)
    assert torch.allclose(out, out_scaled, atol=1e-3)
