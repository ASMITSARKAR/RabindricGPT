"""
src/rgpt/model/mlp.py : SwiGLU Feed-Forward Network without additive bias terms.
Backlog/Phase : I7 / Phase H      Docs : 15_BUILD_DIRECTIVE.md Phase H; 05_MODEL_SPEC.md; DR-3, DR-4
Contract      : x: Tensor -> out: Tensor
Invariants    : SwiGLU formula: (x W_gate * silu(x W_up)) W_down; zero bias vectors; hidden rounded per DR-3
Failure modes : Dimension mismatch between hidden_dim and d_model
Status        : TESTED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward transformation: gate and up projections modulated by SiLU."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass executing SwiGLU gating: w_down(silu(w_gate(x)) * w_up(x))."""
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
