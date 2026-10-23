"""
tests/common/test_seed.py : Tests for deterministic RNG seeding and state round-trip capture/restoration.
"""

import random

import numpy as np
import torch

from src.rgpt.common.seed import get_rng_states, seed_all, set_rng_states


def test_seed_determinism() -> None:
    """Same seed must produce identical numbers across python, numpy, and torch."""
    seed_all(12345)
    py_val1 = random.random()
    np_val1 = np.random.rand()
    torch_val1 = torch.rand(1).item()

    seed_all(12345)
    py_val2 = random.random()
    np_val2 = np.random.rand()
    torch_val2 = torch.rand(1).item()

    assert py_val1 == py_val2
    assert np_val1 == np_val2
    assert torch_val1 == torch_val2


def test_rng_state_round_trip() -> None:
    """Captured RNG states must accurately restore generator state."""
    seed_all(999)
    # Advance some steps
    for _ in range(10):
        _ = random.random()
        _ = np.random.rand()
        _ = torch.rand(5)

    states = get_rng_states()

    # Draw next numbers
    py_next = random.random()
    np_next = np.random.rand()
    torch_next = torch.rand(5)

    # Scramble
    seed_all(888)
    for _ in range(5):
        _ = random.random()

    # Restore states
    set_rng_states(states)

    assert random.random() == py_next
    assert np.random.rand() == np_next
    assert torch.equal(torch.rand(5), torch_next)
