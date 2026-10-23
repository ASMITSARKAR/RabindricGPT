"""
src/rgpt/train/throughput.py : Measurement of training throughput (tokens/s, chars/s) and Model FLOPs Utilization (MFU).
Backlog/Phase : I8, I10 / Phase I, L      Docs : 15_BUILD_DIRECTIVE.md Phase I, L; 08_COMPUTE_BUDGET.md §3
Contract      : compute_mfu(tokens_per_sec, f_tok, peak_flops) -> float
Invariants    : Warmup and torch.compile steps excluded; formula matches PaLM/Chinchilla MFU definitions
Failure modes : Zero elapsed time; missing GPU peak FLOPs specification
Status        : TESTED
"""

from typing import Dict


def calculate_throughput(
    num_tokens: int,
    num_chars: int,
    elapsed_seconds: float,
) -> Dict[str, float]:
    """Calculate token and character processing velocity.

    Args:
        num_tokens: Tokens processed.
        num_chars: Characters processed.
        elapsed_seconds: Elapsed time in seconds.

    Returns:
        Dict containing tokens_per_sec and chars_per_sec.
    """
    if elapsed_seconds <= 0:
        return {"tokens_per_sec": 0.0, "chars_per_sec": 0.0}

    return {
        "tokens_per_sec": num_tokens / elapsed_seconds,
        "chars_per_sec": num_chars / elapsed_seconds,
    }


def calculate_mfu(
    tokens_per_second: float,
    f_tok: float,
    peak_flops: float,
) -> float:
    """Compute Model FLOPs Utilization (MFU).

    Formula: (tokens_per_second * f_tok) / peak_flops

    Args:
        tokens_per_second: Measured training throughput.
        f_tok: FLOPs per token for this model architecture.
        peak_flops: Theoretical peak dense FLOPs of the active GPU.

    Returns:
        Fractional MFU (e.g. 0.32 for 32%).
    """
    if peak_flops <= 0 or tokens_per_second <= 0:
        return 0.0

    return (tokens_per_second * f_tok) / peak_flops
