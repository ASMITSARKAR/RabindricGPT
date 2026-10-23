"""
tests/preflight/test_faults.py : Tests for fault injection mechanism and protection guards.
"""

import os

import pytest

from src.rgpt.preflight.faults import maybe_inject_fault


def test_fault_injection_requires_allow_guard() -> None:
    """Fault injection must remain inactive unless RGPT_ALLOW_FAULT=1 is set."""
    os.environ["RGPT_FAULT"] = "crash@5"

    # Guard not set: must not raise
    if "RGPT_ALLOW_FAULT" in os.environ:
        del os.environ["RGPT_ALLOW_FAULT"]
    maybe_inject_fault("train_step", 5)

    # Guard set: must raise
    os.environ["RGPT_ALLOW_FAULT"] = "1"
    try:
        with pytest.raises(RuntimeError) as exc:
            maybe_inject_fault("train_step", 5)
        assert "[FAULT_INJECTION]" in str(exc.value)
    finally:
        del os.environ["RGPT_ALLOW_FAULT"]
        if "RGPT_FAULT" in os.environ:
            del os.environ["RGPT_FAULT"]
