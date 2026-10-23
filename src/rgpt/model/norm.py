"""
src/rgpt/model/norm.py : Root Mean Square Layer Normalization (RMSNorm) module.
Backlog/Phase : I7 / Phase H      Docs : 15_BUILD_DIRECTIVE.md Phase H; 05_MODEL_SPEC.md
Contract      : x: Tensor -> norm_x: Tensor
Invariants    : Invariant to scaling; no learnable mean bias parameter; float32 calculation for stability
Failure modes : Zero norm input (handled by epsilon); shape mismatch
Status        : TESTED
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization without bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm in fp32 precision and cast back to input dtype."""
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(input_dtype)
