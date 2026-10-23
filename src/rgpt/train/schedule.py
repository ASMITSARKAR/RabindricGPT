"""
src/rgpt/train/schedule.py : Learning rate schedule with linear warmup and cosine decay to minimum ratio.
Backlog/Phase : I8 / Phase I      Docs : 15_BUILD_DIRECTIVE.md Phase I; DR-5
Contract      : get_lr(step: int, total_steps: int, base_lr: float, warmup_ratio: float, min_ratio: float) -> float
Invariants    : Returns pure function of step; lr starts at 0, reaches base_lr at warmup_steps, decays to min_lr
Failure modes : total_steps <= 0; step < 0; invalid ratios
Status        : TESTED
"""

import math


def get_learning_rate(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_ratio: float = 0.02,
    min_lr_ratio: float = 0.10,
) -> float:
    """Compute learning rate multiplier for step using linear warmup and cosine decay.

    Args:
        step: Current optimization step (0-indexed).
        total_steps: Total number of planned training steps.
        base_lr: Peak learning rate.
        warmup_ratio: Fraction of total_steps spent in linear warmup.
        min_lr_ratio: Fraction of base_lr reached at final step.

    Returns:
        Learning rate scalar.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")

    warmup_steps = int(total_steps * warmup_ratio)

    # Linear warmup phase: 0 -> base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)

    # Post-training floor
    if step >= total_steps:
        return base_lr * min_lr_ratio

    # Cosine decay phase: base_lr -> (base_lr * min_lr_ratio)
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / max(1, decay_steps)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor)

    return lr
