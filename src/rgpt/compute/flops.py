"""
src/rgpt/compute/flops.py : FLOPs and compute unit accounting formulas matching 08_COMPUTE_BUDGET.md anchors.
Backlog/Phase : I10 / Phase L      Docs : 15_BUILD_DIRECTIVE.md §2, Phase L; 08_COMPUTE_BUDGET.md §2
Contract      : calculate_f_tok(n_ne, v, d, l, s) -> float; calculate_units(c, flops_per_unit) -> float
Invariants    : Matches 08 §2 anchors: F_tok(10M, byte) = 149.2M, F_tok(10M, bpe16) = 108.5M within 3 sig digits
Failure modes : Zero or negative parameter count; missing GPU regime
Status        : TESTED
"""


def calculate_f_tok(
    n_ne: int,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    s_tokens: int,
) -> float:
    """Calculate training FLOPs per token.

    Formula:
        F_tok = 6 * (N_ne + V * d) + 12 * L * d * s

    Args:
        n_ne: Non-embedding parameter count.
        vocab_size: Vocabulary size (V).
        d_model: Model hidden dimension (d).
        n_layers: Number of transformer layers (L).
        s_tokens: Tokens per window (s).

    Returns:
        FLOPs per token as a float.
    """
    if n_ne <= 0 or d_model <= 0 or n_layers <= 0 or s_tokens <= 0:
        raise ValueError("Dimensions and parameter counts must be strictly positive.")

    # 6 * (N_ne + V * d) + 12 * L * d * s
    param_term = 6.0 * (n_ne + vocab_size * d_model)
    attn_term = 12.0 * n_layers * d_model * s_tokens
    return float(param_term + attn_term)


def calculate_run_flops(
    f_tok: float,
    d_bytes: float,
    t_bytes_per_tok: float,
) -> float:
    """Calculate total training run FLOPs C = F_tok * (D_bytes / T).

    Args:
        f_tok: FLOPs per token.
        d_bytes: Total dataset volume in bytes.
        t_bytes_per_tok: Compression rate T in bytes per token.

    Returns:
        Total run FLOPs C.
    """
    if t_bytes_per_tok <= 0:
        raise ValueError("Compression rate T must be strictly positive.")

    d_tok = d_bytes / t_bytes_per_tok
    return float(f_tok * d_tok)


def calculate_units(
    c_flops: float,
    flops_per_unit: float = 2.2e16,
) -> float:
    """Convert total run FLOPs to Colab compute units.

    Formula:
        units = C / FLOPs_per_unit (default 2.2e16 central planning prior)
    """
    if flops_per_unit <= 0:
        raise ValueError("FLOPs per unit must be strictly positive.")
    return float(c_flops / flops_per_unit)
