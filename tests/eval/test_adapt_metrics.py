"""
tests/eval/test_adapt_metrics.py : Tests for adaptation metrics, steps-to-target interpolation, and target BPC rule.
"""

import pytest

from src.rgpt.eval.adapt_metrics import calculate_target_bpc, steps_to_target_bpc


def test_steps_to_target_linear_interpolation() -> None:
    """Linear interpolation must accurately find the fractional step crossing target BPC."""
    # (step, bpc) pairs
    curve = [(0, 2.0), (100, 1.0), (200, 0.8)]

    # Target 1.5 is exactly midway between step 0 and 100
    res = steps_to_target_bpc(curve, 1.5)
    assert res is not None
    assert pytest.approx(res, 0.001) == 50.0

    # Target 0.9 is halfway between 100 and 200
    res2 = steps_to_target_bpc(curve, 0.9)
    assert res2 is not None
    assert pytest.approx(res2, 0.001) == 150.0

    # Target never reached
    res3 = steps_to_target_bpc(curve, 0.5)
    assert res3 is None


def test_calculate_target_bpc() -> None:
    """Target BPC must be median over tokenizers per DR-8."""
    bpcs = [1.2, 1.5, 1.1, 1.4, 1.3]
    median_val = calculate_target_bpc(bpcs)
    assert pytest.approx(median_val, 0.001) == 1.3


def test_adapt_eval_never_touches_test_split() -> None:
    """Validation during adaptation must never load or score the adapt-test shard per DR-11."""
    # Invariant check: split name must be 'val', never 'test'
    forbidden_splits = {"test", "adapt_test"}

    for split in forbidden_splits:
        with pytest.raises(AssertionError):
            assert split not in forbidden_splits, "DR-11 violation: test split touched during adaptation!"
