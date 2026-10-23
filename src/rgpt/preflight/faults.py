"""
src/rgpt/preflight/faults.py : Fault injection harness (RGPT_FAULT) for Level P4 Colab-parity resilience testing.
Backlog/Phase : Phase P      Docs : 15_BUILD_DIRECTIVE.md Phase P (Level P4)
Contract      : maybe_inject_fault(hook_name: str, step: int) -> None
Invariants    : Only active when RGPT_ALLOW_FAULT=1; runner refuses real runs if RGPT_FAULT is set
Failure modes : Fault triggers in production without RGPT_ALLOW_FAULT guard
Status        : TESTED
"""

import os
from pathlib import Path


def maybe_inject_fault(hook_point: str, step: int) -> None:
    """Inject simulated failure if matching RGPT_FAULT is configured in environment.

    Supported faults:
        - crash@<k>: raise RuntimeError at step k
        - kill@<k>: hard os._exit(1) at step k
        - nan@<k>: raise ValueError with NaN loss at step k
        - corrupt_ckpt: corrupt the latest checkpoint file

    Args:
        hook_point: Current execution hook point.
        step: Current training or runner step.
    """
    # Guard: fault injection must never activate in real runs
    if os.environ.get("RGPT_ALLOW_FAULT") != "1":
        return

    fault_spec = os.environ.get("RGPT_FAULT", "")
    if not fault_spec:
        return

    # Check step-based faults
    if "@" in fault_spec:
        action, target_step_str = fault_spec.split("@", 1)
        try:
            target_step = int(target_step_str)
        except ValueError:
            return

        if step == target_step:
            if action == "crash":
                raise RuntimeError(f"[FAULT_INJECTION] Simulated crash at step {step}")
            elif action == "kill":
                # Simulated kill
                os._exit(1)
            elif action == "nan":
                raise ValueError(f"[FAULT_INJECTION] Simulated NaN loss divergence at step {step}")

    elif fault_spec == "corrupt_ckpt" and hook_point == "checkpoint_saved":
        ckpt_path = os.environ.get("RGPT_LAST_CKPT")
        if ckpt_path and Path(ckpt_path).exists():
            with open(ckpt_path, "wb") as f:
                f.write(b"CORRUPTED_BYTES")
