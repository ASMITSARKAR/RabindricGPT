"""
tests/train/test_throughput.py : Tests for throughput calculation and Model FLOPs Utilization (MFU).
"""

import pytest

from src.rgpt.train.throughput import calculate_mfu, calculate_throughput


def test_throughput_and_mfu_calculation() -> None:
    """Throughput and MFU must match standard formula excluding warmup steps."""
    tp = calculate_throughput(num_tokens=10000, num_chars=25000, elapsed_seconds=2.0)
    assert tp["tokens_per_sec"] == pytest.approx(5000.0)
    assert tp["chars_per_sec"] == pytest.approx(12500.0)

    # L4 theoretical peak: 121 TFLOPs = 1.21e14 FLOPs/s
    tokens_per_sec = 5000.0
    f_tok = 1.0e8  # 100M FLOPs/token
    peak_flops = 1.21e14

    mfu = calculate_mfu(tokens_per_sec, f_tok, peak_flops)
    expected_mfu = (5000.0 * 1.0e8) / 1.21e14
    assert mfu == pytest.approx(expected_mfu)
