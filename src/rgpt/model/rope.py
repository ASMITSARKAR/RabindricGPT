"""
src/rgpt/model/rope.py : Rotary Position Embedding (RoPE) precomputation and application to query and key tensors.
Backlog/Phase : I7 / Phase H      Docs : 15_BUILD_DIRECTIVE.md Phase H; 05_MODEL_SPEC.md
Contract      : apply_rope(q, k, seq_len) -> (q_rot, k_rot)
Invariants    : Relative position property holds; only applied to q and k; shapes preserved
Failure modes : Sequence length exceeds precomputed max_seq_len; tensor dimension mismatch
Status        : TESTED
"""

from typing import Tuple

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Precomputed Rotary Position Embedding (RoPE) frequencies and cache."""

    def __init__(
        self, dim: int, max_seq_len: int = 3072, theta: float = 10000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim // 2]

        # Duplicate along last dimension to match full dim
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = torch.cat((cos, cos), dim=-1)  # [max_seq_len, dim]
        sin = torch.cat((sin, sin), dim=-1)  # [max_seq_len, dim]

        # Register non-persistent buffers: shape [1, 1, max_seq_len, dim]
        self.register_buffer("cos_cached", cos.unsqueeze(0).unsqueeze(1), persistent=False)
        self.register_buffer("sin_cached", sin.unsqueeze(0).unsqueeze(1), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary transformation to query and key projections.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len, head_dim).
            k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim).
            seq_len: Current sequence length.

        Returns:
            Tuple of rotated (q, k) tensors.
        """
        if seq_len > self.cos_cached.shape[2]:
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
            t = torch.arange(seq_len, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            cos_new = torch.cos(freqs)
            sin_new = torch.sin(freqs)
            cos_new = torch.cat((cos_new, cos_new), dim=-1).unsqueeze(0).unsqueeze(1)
            sin_new = torch.cat((sin_new, sin_new), dim=-1).unsqueeze(0).unsqueeze(1)
            self.register_buffer("cos_cached", cos_new, persistent=False)
            self.register_buffer("sin_cached", sin_new, persistent=False)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)

        return q_rot, k_rot
