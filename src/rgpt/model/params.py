"""
src/rgpt/model/params.py : Analytic parameter counting functions for exact total and non-embedding model parameters.
Backlog/Phase : I7 / Phase H      Docs : 15_BUILD_DIRECTIVE.md Phase H; 08_COMPUTE_BUDGET.md §2; DR-3
Contract      : config: ModelConfig -> (total_params: int, non_embedding_params: int)
Invariants    : count_parameters(config) == sum(p.numel() for p in model.parameters()) exactly
Failure modes : Discrepancy between analytic formula and actual PyTorch parameter tensors
Status        : TESTED
"""

from typing import Any, Dict, Set, Tuple

import torch.nn as nn

from src.rgpt.model.config import ModelConfig


def count_parameters_analytic(config: ModelConfig) -> Tuple[int, int]:
    """Calculate exact theoretical parameter count from ModelConfig dimensions.

    Returns:
        Tuple of (total_parameters, non_embedding_parameters).
    """
    d = config.d_model
    l_layers = config.n_layers
    hidden = config.hidden_dim
    head_dim = config.head_dim
    n_heads = config.n_heads
    n_kv = config.n_kv_heads
    v = config.vocab_size

    # Per-layer attention: Q, K, V, Out
    q_params = d * (n_heads * head_dim)
    k_params = d * (n_kv * head_dim)
    v_params = d * (n_kv * head_dim)
    out_params = (n_heads * head_dim) * d
    attn_layer = q_params + k_params + v_params + out_params

    # Per-layer SwiGLU MLP: Gate, Up, Down
    mlp_layer = 3 * d * hidden

    # Per-layer LayerNorms: attn_norm (d), mlp_norm (d)
    norms_layer = 2 * d

    # Layer total
    per_layer = attn_layer + mlp_layer + norms_layer

    # Final norm (d)
    final_norm = d

    # Total non-embedding
    non_embedding_params = (l_layers * per_layer) + final_norm

    # Embedding: V * d (tied with lm_head, so only counted once)
    embedding_params = v * d
    total_params = non_embedding_params + embedding_params

    return total_params, non_embedding_params


def count_parameters_empirical(model: nn.Module) -> Tuple[int, int]:
    """Count parameters directly from PyTorch module tensors.

    Returns:
        Tuple of (total_parameters, non_embedding_params).
    """
    # Use set to avoid double-counting tied weights (lm_head.weight is w_token.weight)
    unique_params: Set[nn.Parameter] = set(model.parameters())
    total_params = sum(p.numel() for p in unique_params)

    # Non-embedding parameters exclude w_token.weight
    embed_param = getattr(model, "w_token", None)
    embed_weight = embed_param.weight if embed_param is not None else None

    non_embedding_params = sum(
        p.numel() for p in unique_params if p is not embed_weight
    )

    return total_params, non_embedding_params


def compare_parameter_counts(config: ModelConfig, model: nn.Module) -> Dict[str, Any]:
    """Verify that analytic and empirical parameter counts match exactly."""
    analytic_tot, analytic_ne = count_parameters_analytic(config)
    empirical_tot, empirical_ne = count_parameters_empirical(model)

    tot_diff = empirical_tot - analytic_tot
    ne_diff = empirical_ne - analytic_ne

    matches = (tot_diff == 0) and (ne_diff == 0)

    return {
        "matches": matches,
        "analytic_total": analytic_tot,
        "empirical_total": empirical_tot,
        "total_diff": tot_diff,
        "analytic_non_embed": analytic_ne,
        "empirical_non_embed": empirical_ne,
        "non_embed_diff": ne_diff,
    }
