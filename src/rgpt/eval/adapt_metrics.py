"""
src/rgpt/eval/adapt_metrics.py : Adaptation metrics computation: steps-to-target BPC and transfer gain.
Backlog/Phase : I13 / Phase K      Docs : 15_BUILD_DIRECTIVE.md Phase K; DR-8; 01 §1 (H5)
Contract      : val_bpc_curve, target_bpc: float -> steps_to_target: float
Invariants    : Target BPC computed per DR-8 (median over tokenizers at that size); linear interpolation
Failure modes : Target BPC never crossed during adaptation epochs
Status        : TESTED
"""

import statistics
from typing import List, Optional, Tuple


def calculate_target_bpc(final_val_bpcs: List[float]) -> float:
    """Compute target BPC as median over tokenizers per DR-8."""
    if not final_val_bpcs:
        raise ValueError("Cannot calculate target BPC from empty list.")
    return float(statistics.median(final_val_bpcs))


def steps_to_target_bpc(
    step_bpc_pairs: List[Tuple[int, float]],
    target_bpc: float,
) -> Optional[float]:
    """Find the step count at which validation BPC crossed below or equal to target_bpc via linear interpolation.

    Args:
        step_bpc_pairs: Sorted list of (step, bpc) tuples.
        target_bpc: The threshold BPC to achieve.

    Returns:
        Interpolated step number as float, or None if the threshold was never reached.
    """
    if not step_bpc_pairs:
        return None

    # Check if starting point already satisfies target
    if step_bpc_pairs[0][1] <= target_bpc:
        return float(step_bpc_pairs[0][0])

    for i in range(1, len(step_bpc_pairs)):
        prev_step, prev_bpc = step_bpc_pairs[i - 1]
        curr_step, curr_bpc = step_bpc_pairs[i]

        if prev_bpc > target_bpc >= curr_bpc:
            # Linear interpolation
            delta_bpc = prev_bpc - curr_bpc
            if delta_bpc == 0:
                return float(curr_step)
            fraction = (prev_bpc - target_bpc) / delta_bpc
            return prev_step + fraction * (curr_step - prev_step)

    return None
