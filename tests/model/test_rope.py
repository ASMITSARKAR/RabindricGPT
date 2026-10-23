"""
tests/model/test_rope.py : Tests for Rotary Position Embedding relative-position preservation.
"""

import pytest
import torch

from src.rgpt.model.rope import RotaryEmbedding


def test_rope_relative_position_property() -> None:
    """Dot product of rotated q and k must depend purely on relative position (t - s)."""
    dim = 64
    seq_len = 100
    rope = RotaryEmbedding(dim=dim, max_seq_len=200, theta=10000.0)

    # Identical query and key vector placed at different positions
    q_vec = torch.randn(dim)
    k_vec = torch.randn(dim)

    # Construct batch of sequences
    # Sequence 1: q at position 10, k at position 5 (diff = 5)
    # Sequence 2: q at position 20, k at position 15 (diff = 5)
    q = torch.zeros(1, 1, seq_len, dim)
    k = torch.zeros(1, 1, seq_len, dim)

    q[0, 0, 10] = q_vec
    k[0, 0, 5] = k_vec
    q[0, 0, 20] = q_vec
    k[0, 0, 15] = k_vec

    q_rot, k_rot = rope(q, k, seq_len=seq_len)

    dot_1 = (q_rot[0, 0, 10] * k_rot[0, 0, 5]).sum().item()
    dot_2 = (q_rot[0, 0, 20] * k_rot[0, 0, 15]).sum().item()

    assert dot_1 == pytest.approx(dot_2, rel=1e-3, abs=1e-3)
