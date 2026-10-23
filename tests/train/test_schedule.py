"""
tests/train/test_schedule.py : Tests for linear warmup and cosine decay learning rate schedule.
"""

import pytest

from src.rgpt.train.schedule import get_learning_rate


def test_lr_schedule_warmup_and_cosine_values() -> None:
    """LR must scale linearly during warmup and follow cosine curve to min_lr."""
    total_steps = 1000
    base_lr = 1e-3
    warmup_ratio = 0.05  # 50 warmup steps
    min_lr_ratio = 0.10  # 1e-4

    # Warmup start
    lr_0 = get_learning_rate(0, total_steps, base_lr, warmup_ratio, min_lr_ratio)
    assert lr_0 == pytest.approx(base_lr * 1 / 50)

    # Warmup midpoint
    lr_mid_warmup = get_learning_rate(24, total_steps, base_lr, warmup_ratio, min_lr_ratio)
    assert lr_mid_warmup == pytest.approx(base_lr * 25 / 50)

    # Warmup peak (step 49 is last warmup step, step 50 is base_lr)
    lr_peak = get_learning_rate(50, total_steps, base_lr, warmup_ratio, min_lr_ratio)
    assert lr_peak == pytest.approx(base_lr)

    # Midpoint of decay
    mid_step = 50 + (total_steps - 50) // 2
    lr_mid = get_learning_rate(mid_step, total_steps, base_lr, warmup_ratio, min_lr_ratio)
    assert lr_mid < base_lr
    assert lr_mid > base_lr * min_lr_ratio

    # Final step
    lr_final = get_learning_rate(total_steps, total_steps, base_lr, warmup_ratio, min_lr_ratio)
    assert lr_final == pytest.approx(base_lr * min_lr_ratio)
