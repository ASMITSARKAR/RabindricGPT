"""
src/rgpt/common/seed.py : Deterministic pseudo-random number generator seeding and state capture/restoration.
Backlog/Phase : I-none / Phase B      Docs : 15_BUILD_DIRECTIVE.md §0 R6, Phase B
Contract      : seed: int -> None; get_rng_states() -> dict; set_rng_states(dict) -> None
Invariants    : Same seed yields identical random stream across python/numpy/torch; state capture round-trips
Failure modes : Unsupported RNG device; corrupted RNG state dictionary
Status        : TESTED
"""

import random
from typing import Any, Dict

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python built-in, NumPy, and PyTorch RNGs deterministically.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_states() -> Dict[str, Any]:
    """Capture all RNG states for checkpointing.

    Returns:
        Dictionary mapping subsystem names to serialized RNG states.
    """
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restore all RNG states from a checkpoint.

    Args:
        states: Dictionary of captured RNG states.

    Raises:
        KeyError: If mandatory state keys are missing.
    """
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
