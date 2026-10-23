"""
tests/eval/test_bootstrap.py : Tests for bootstrap resampling confidence intervals.
"""

import math

import numpy as np

from src.rgpt.eval.bootstrap import compute_bootstrap_ci, compute_stratified_breakdown


def test_bootstrap_ci_coverage_on_synthetic_data() -> None:
    """Bootstrap 95% CI must bracket true population mean on synthetic samples."""
    np.random.seed(42)
    n_chunks = 100
    true_mean_bpc = 1.35
    ln2 = math.log(2.0)

    # Generate synthetic chunk characters and nll
    chunk_chars = [1024] * n_chunks
    # nll_nats = bpc * chars * ln2
    sampled_bpcs = np.random.normal(true_mean_bpc, 0.05, n_chunks)
    chunk_nll = [float(bpc * 1024 * ln2) for bpc in sampled_bpcs]

    mean_bpc, ci_lower, ci_upper = compute_bootstrap_ci(
        chunk_nll, chunk_chars, n_resamples=500, seed=42
    )

    assert ci_lower <= true_mean_bpc <= ci_upper
    assert ci_lower < mean_bpc < ci_upper


def test_stratified_breakdown() -> None:
    """Stratified breakdown must compute separate group estimates."""
    records = [
        {"work_id": "w1", "nll_nats": 100.0, "n_chars": 100},
        {"work_id": "w1", "nll_nats": 110.0, "n_chars": 100},
        {"work_id": "w2", "nll_nats": 150.0, "n_chars": 100},
        {"work_id": "w2", "nll_nats": 160.0, "n_chars": 100},
    ]
    breakdown = compute_stratified_breakdown(records, n_resamples=100)
    assert "w1" in breakdown
    assert "w2" in breakdown
    assert breakdown["w1"]["mean_bpc"] < breakdown["w2"]["mean_bpc"]
