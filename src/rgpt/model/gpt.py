"""
src/rgpt/model/gpt.py : RabindricGPT decoder-only causal language model with tied embeddings.
Backlog/Phase : I7 / Phase H      Docs : 15_BUILD_DIRECTIVE.md Phase H; 05_MODEL_SPEC.md; DR-4
Contract      : input_ids: Tensor -> logits: Tensor (batch, seq_len, vocab_size)
Invariants    : Head weights tied to embedding weights (w_head.weight is w_token.weight); no bias vectors
Failure modes : Input ID >= vocab_size; context length > max_seq_len; NaN in forward
Status        : TESTED
"""

from typing import Optional

import torch
import torch.nn as nn

from src.rgpt.model.block import TransformerBlock
from src.rgpt.model.config import ModelConfig
from src.rgpt.model.norm import RMSNorm
from src.rgpt.model.rope import RotaryEmbedding


class RabindricGPT(nn.Module):
    """RabindricGPT decoder-only transformer language model architecture."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.w_token = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                hidden_dim=config.hidden_dim,
                head_dim=config.head_dim,
                rms_norm_eps=config.rms_norm_eps,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.w_token.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute unnormalized log probabilities (logits) for input tokens.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len).
            attention_mask: Optional attention mask.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        x = self.w_token(input_ids)
        for block in self.blocks:
            x = block(x, self.rope, attention_mask=attention_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


GPT = RabindricGPT

