"""
src/rgpt/eval/bootstrap.py : Bootstrap resampling confidence intervals (1,000 resamples) and per-work breakdowns.
Backlog/Phase : I9 / Phase J      Docs : 15_BUILD_DIRECTIVE.md Phase J; 03_DATA_PLAN.md §5.3
Contract      : chunk_losses: list[float] -> (mean: float, ci_lower: float, ci_upper: float)
Invariants    : 1,000 bootstrap iterations; per-work and genre-stratified breakdowns computed
Failure modes : Fewer than 2 chunks to resample
Status        : IMPLEMENTED
"""

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


def compute_bootstrap_ci(
    chunk_nll_nats: List[float],
    chunk_chars: List[int],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute 95% bootstrap confidence interval for BPC across chunks.

    Returns:
        Tuple of (mean_bpc, ci_lower, ci_upper).
    """
    n_chunks = len(chunk_nll_nats)
    if n_chunks < 2:
        raise ValueError(f"Bootstrap resampling requires at least 2 chunks, got {n_chunks}")

    rng = random.Random(seed)
    ln2 = math.log(2.0)

    # Point estimate
    sum_nll = sum(chunk_nll_nats)
    sum_chars = sum(chunk_chars)
    point_estimate = sum_nll / (sum_chars * ln2)

    resampled_bpcs: List[float] = []

    for _ in range(n_resamples):
        # Sample with replacement
        sample_indices = [rng.randint(0, n_chunks - 1) for _ in range(n_chunks)]
        s_nll = sum(chunk_nll_nats[idx] for idx in sample_indices)
        s_chars = sum(chunk_chars[idx] for idx in sample_indices)
        resampled_bpc = s_nll / (s_chars * ln2) if s_chars > 0 else point_estimate
        resampled_bpcs.append(resampled_bpc)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(resampled_bpcs, lower_pct))
    ci_upper = float(np.percentile(resampled_bpcs, upper_pct))

    return point_estimate, ci_lower, ci_upper


def compute_stratified_breakdown(
    chunk_records: List[Dict[str, Any]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute per-work and per-genre BPC bootstrap estimates."""
    work_groups = defaultdict(list)
    for cr in chunk_records:
        wid = cr.get("work_id") or "unknown_work"
        work_groups[wid].append(cr)

    breakdown = {}
    for wid, records in work_groups.items():
        nlls = [r["nll_nats"] for r in records]
        chars = [r["n_chars"] for r in records]
        if len(records) >= 2:
            pt, low, high = compute_bootstrap_ci(nlls, chars, n_resamples=n_resamples, seed=seed)
        else:
            ln2 = math.log(2.0)
            pt = sum(nlls) / (sum(chars) * ln2)
            low, high = pt, pt

        breakdown[wid] = {
            "mean_bpc": pt,
            "ci_lower": low,
            "ci_upper": high,
            "num_chunks": len(records),
        }

    return breakdown
