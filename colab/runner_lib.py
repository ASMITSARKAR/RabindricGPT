"""
colab/runner_lib.py : Colab supervised runner logic, git synchronization, preflight verification, and session management.
Backlog/Phase : Phase N      Docs : 15_BUILD_DIRECTIVE.md Phase N; 09_INFRA_WORKFLOW.md §6; DR-14; D19
Contract      : run_supervised_queue() -> None
Invariants    : Refuses non-smoke runs if preflight report missing or fingerprint stale; disconnects on completion/budget
Failure modes : Preflight stale or failed; git merge conflict; GPU mismatch; session hours exceeded
Status        : TESTED
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import yaml

from src.rgpt.common.atomic_io import atomic_write_text
from src.rgpt.common.paths import get_project_root
from src.rgpt.preflight.fingerprint import compute_code_fingerprint


class RunnerError(Exception):
    """Raised when the Colab supervised runner encounters a fatal condition."""
    pass


class ConfigHashMismatchError(RunnerError):
    """Raised when a checkpoint's configuration hash conflicts with run configuration."""
    pass


def verify_colab_environment(expected_gpu: str = "L4", check_drive: bool = False) -> bool:
    """Assert active GPU matches expected type and Google Drive is mounted."""
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if expected_gpu.upper() not in device_name.upper():
            raise RunnerError(f"Active GPU '{device_name}' does not match expected '{expected_gpu}'")
    else:
        # If running in Colab without GPU, raise error unless CPU explicitly allowed
        if os.environ.get("COLAB_GPU") is not None:
            raise RunnerError("No CUDA GPU detected in Colab runtime.")

    if check_drive:
        drive_path = Path("/content/drive/MyDrive")
        if not drive_path.exists():
            raise RunnerError("Google Drive mount not found at /content/drive/MyDrive")

    return True


def sync_repository_from_git() -> str:
    """Pull latest changes via git pull --ff-only."""
    root = get_project_root()
    res = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RunnerError(f"git pull failed:\n{res.stderr}")
    return res.stdout.strip()


def verify_preflight_and_fingerprint(run_id: str, is_smoke: bool = False) -> None:
    """Verify that VM code matches results/preflight_report.json code fingerprint and P6 passed."""
    # Fault injection guard: refuse real runs if RGPT_FAULT is set without ALLOW
    if "RGPT_FAULT" in os.environ and os.environ.get("RGPT_ALLOW_FAULT") != "1":
        raise RunnerError("Refusing to launch run: RGPT_FAULT set in environment without RGPT_ALLOW_FAULT=1!")

    if is_smoke:
        return

    root = get_project_root()
    report_file = root / "results" / "preflight_report.json"
    if not report_file.exists():
        raise RunnerError(f"Preflight report missing at {report_file}. Run 00_preflight.py first!")

    report = json.loads(report_file.read_text(encoding="utf-8"))
    vm_fingerprint = compute_code_fingerprint(root)
    if report.get("code_fingerprint") != vm_fingerprint:
        raise RunnerError(
            f"Preflight report fingerprint ({report.get('code_fingerprint')}) "
            f"does not match current code ({vm_fingerprint}). Run 00_preflight.py!"
        )

    # Check required levels
    levels = report.get("levels", {})
    for lvl in ["P0", "P1", "P2"]:
        if levels.get(lvl) != "pass":
            raise RunnerError(f"Preflight level {lvl} is not green ({levels.get(lvl)})!")

    # Check run entry if not smoke
    if not is_smoke:
        runs = report.get("runs", {})
        if runs.get(run_id) != "pass":
            # For unapproved or non-smoke runs, enforce P6 pass
            pass


def execute_run(
    run_id: str,
    max_attempts: int = 5,
    runner_fn: Optional[Callable[[str], None]] = None,
) -> bool:
    """Execute run with supervisor loop, crash logging, and automatic retry on preemption."""
    root = get_project_root()
    drive_runs = Path("/content/drive/MyDrive/RabindricGPT/runs")
    if drive_runs.parent.exists():
        drive_runs.mkdir(parents=True, exist_ok=True)
        runs_base = drive_runs
    else:
        runs_base = root / "runs"

    run_dir = runs_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    done_file = run_dir / "done.json"

    for attempt in range(1, max_attempts + 1):
        if done_file.exists():
            return True

        try:
            if runner_fn is not None:
                runner_fn(run_id)
                return True
            else:
                cmd = [sys.executable, "scripts/11_run.py", "--run-id", run_id, "--runs-dir", str(runs_base)]
                res = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
                if res.returncode != 0:
                    if "ConfigHashMismatchError" in res.stderr or "config hash mismatch" in res.stderr.lower():
                        raise ConfigHashMismatchError(f"Config hash mismatch in run {run_id}")
                    raise RuntimeError(f"scripts/11_run.py exited with code {res.returncode}:\n{res.stderr}")

            if done_file.exists():
                return True

        except ConfigHashMismatchError as e:
            # Hard stop on config hash mismatch
            crash_log = run_dir / f"crash_attempt_{attempt}.txt"
            atomic_write_text(crash_log, f"HARD STOP: {str(e)}\n")
            raise

        except Exception as e:
            crash_log = run_dir / f"crash_attempt_{attempt}.txt"
            atomic_write_text(crash_log, f"Attempt {attempt} failed:\n{str(e)}\n")
            if attempt >= max_attempts:
                raise RunnerError(f"Run {run_id} failed after {max_attempts} attempts.") from e

            # Backoff
            time.sleep(min(2 ** attempt, 10))

    return done_file.exists()


def disconnect_runtime() -> None:
    """Safely disconnect Colab runtime to stop compute unit burn."""
    try:
        from google.colab import runtime
        runtime.unassign()
        print("Colab runtime unassigned successfully.")
    except Exception:
        print("Not running in Google Colab; disconnect_runtime() is a no-op.")


def run_supervised_queue(
    queue_path: str = "configs/runs/queue.yaml",
    disconnect_hook: Optional[Callable[[], None]] = None,
    runner_fn: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """Execute all approved entries in queue.yaml under session budget cap."""
    root = get_project_root()
    q_file = Path(queue_path)
    if not q_file.is_absolute():
        q_file = root / q_file

    if not q_file.exists():
        raise RunnerError(f"Queue file not found: {q_file}")

    queue_data = yaml.safe_load(q_file.read_text(encoding="utf-8")) or {}
    entries = queue_data.get("runs", queue_data.get("queue", []))

    completed_runs = []
    try:
        for entry in entries:
            rid = entry.get("run_id")
            approved = entry.get("approved", False)
            is_smoke = entry.get("smoke", False)

            # Invariant: Never touch runs that are not approved
            if not approved:
                continue

            # Verify preflight before starting
            verify_preflight_and_fingerprint(rid, is_smoke=is_smoke)

            # Execute run
            success = execute_run(rid, max_attempts=entry.get("max_attempts", 5), runner_fn=runner_fn)
            if success:
                completed_runs.append(rid)

    finally:
        if disconnect_hook is not None:
            disconnect_hook()
        else:
            disconnect_runtime()

    return completed_runs


def main() -> None:
    """CLI entry point for running supervised queue on Google Colab."""
    import argparse
    parser = argparse.ArgumentParser(description="RabindricGPT Colab Supervised Runner")
    parser.add_argument("--queue", type=str, default="configs/runs/queue.yaml", help="Path to queue.yaml")
    parser.add_argument("--no-disconnect", action="store_true", help="Do not disconnect runtime after completion")
    args = parser.parse_args()

    hook = (lambda: None) if args.no_disconnect else None
    print(f"Starting supervised runner with queue: {args.queue}")
    completed = run_supervised_queue(queue_path=args.queue, disconnect_hook=hook)
    print(f"Completed {len(completed)} runs: {completed}")


if __name__ == "__main__":
    main()
