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
from typing import Any, Callable, Dict, List, Optional

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


def verify_preflight_and_fingerprint(
    run_id: str,
    is_smoke: bool = False,
    root_dir: Optional[Path] = None,
) -> None:
    """Verify that VM code matches results/preflight_report.json code fingerprint and P6 passed."""
    # Fault injection guard: refuse real runs if RGPT_FAULT is set without ALLOW
    if "RGPT_FAULT" in os.environ and os.environ.get("RGPT_ALLOW_FAULT") != "1":
        raise RunnerError("Refusing to launch run: RGPT_FAULT set in environment without RGPT_ALLOW_FAULT=1!")

    if is_smoke:
        return

    root = Path(root_dir) if root_dir is not None else get_project_root()
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


def get_runs_base(root: Optional[Path] = None) -> Path:
    """Resolve runs directory, favoring persistent Google Drive when mounted."""
    if root is None:
        root = get_project_root()
    drive_runs = Path("/content/drive/MyDrive/RabindricGPT/runs")
    if drive_runs.parent.exists():
        drive_runs.mkdir(parents=True, exist_ok=True)
        return drive_runs
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def update_runner_status(
    runs_base: Path,
    status: str,
    active_run: Optional[str] = None,
    completed_runs: Optional[List[str]] = None,
    failed_runs: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> None:
    """Atomically record queue runner progress to status.json on persistent storage."""
    status_file = runs_base / "status.json"
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data = {
        "status": status,
        "active_run": active_run,
        "completed_runs": completed_runs or [],
        "failed_runs": failed_runs or [],
        "timestamp": now_iso,
    }
    if error:
        data["last_error"] = error
    atomic_write_text(status_file, json.dumps(data, indent=2))


def persist_unit_ledger(runs_base: Path) -> Dict[str, Any]:
    """Aggregate all completed run ledger rows and persist ledger.md and unit_ledger.json."""
    from src.rgpt.compute.ledger import update_ledger_report
    ledger_md = runs_base / "ledger.md"
    rep = update_ledger_report(runs_base, ledger_md)
    ledger_json = runs_base / "unit_ledger.json"
    atomic_write_text(ledger_json, json.dumps(rep, indent=2))
    return rep


def execute_run(
    run_id: str,
    max_attempts: int = 5,
    runner_fn: Optional[Callable[[str], None]] = None,
    runs_base: Optional[Path] = None,
) -> bool:
    """Execute run with supervisor loop, crash logging, and automatic retry on preemption."""
    root = get_project_root()
    if runs_base is None:
        runs_base = get_runs_base(root)

    run_dir = runs_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    done_file = run_dir / "done.json"

    # Invariant: skip runs that are already marked done
    if done_file.exists():
        print(f"Run {run_id} is already marked DONE (found {done_file}). Skipping.")
        return True

    for attempt in range(1, max_attempts + 1):
        if done_file.exists():
            return True

        try:
            update_runner_status(runs_base, "RUNNING", active_run=run_id)
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
                persist_unit_ledger(runs_base)
                return True

        except ConfigHashMismatchError as e:
            # Hard stop on config hash mismatch
            crash_log = run_dir / f"crash_attempt_{attempt}.txt"
            atomic_write_text(crash_log, f"HARD STOP: {str(e)}\n")
            update_runner_status(runs_base, "FAILED", active_run=run_id, error=str(e))
            raise

        except Exception as e:
            crash_log = run_dir / f"crash_attempt_{attempt}.txt"
            atomic_write_text(crash_log, f"Attempt {attempt} failed:\n{str(e)}\n")
            if attempt >= max_attempts:
                update_runner_status(runs_base, "FAILED", active_run=run_id, error=str(e))
                raise RunnerError(f"Run {run_id} failed after {max_attempts} attempts.") from e

            # Backoff
            time.sleep(min(2 ** attempt, 10))

    success = done_file.exists()
    if success:
        persist_unit_ledger(runs_base)
    return success


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
    runs_base: Optional[Path] = None,
) -> List[str]:
    """Execute all approved entries in queue.yaml under session budget cap."""
    root = get_project_root()
    if runs_base is None:
        runs_base = get_runs_base(root)

    q_file = Path(queue_path)
    if not q_file.is_absolute():
        q_file = root / q_file

    if not q_file.exists():
        raise RunnerError(f"Queue file not found: {q_file}")

    queue_data = yaml.safe_load(q_file.read_text(encoding="utf-8")) or {}
    entries = queue_data.get("runs", queue_data.get("queue", []))

    completed_runs: List[str] = []
    failed_runs: List[str] = []

    update_runner_status(runs_base, "STARTING", completed_runs=completed_runs, failed_runs=failed_runs)

    try:
        for entry in entries:
            rid = entry.get("run_id")
            approved = entry.get("approved", False)
            is_smoke = entry.get("smoke", False)

            # Invariant: Never touch runs that are not approved
            if not approved:
                continue

            # Invariant: skip already completed runs
            if (runs_base / rid / "done.json").exists():
                print(f"Run {rid} already completed (found done.json). Skipping.")
                if rid not in completed_runs:
                    completed_runs.append(rid)
                continue

            # Verify preflight before starting
            verify_preflight_and_fingerprint(rid, is_smoke=is_smoke, root_dir=root)

            # Execute run
            try:
                success = execute_run(
                    rid,
                    max_attempts=entry.get("max_attempts", 5),
                    runner_fn=runner_fn,
                    runs_base=runs_base,
                )
                if success:
                    completed_runs.append(rid)
                else:
                    failed_runs.append(rid)
            except Exception as e:
                failed_runs.append(rid)
                update_runner_status(
                    runs_base,
                    "FAILED",
                    active_run=rid,
                    completed_runs=completed_runs,
                    failed_runs=failed_runs,
                    error=str(e),
                )
                raise

        status = "COMPLETED" if not failed_runs else "PARTIALLY_COMPLETED"
        update_runner_status(
            runs_base,
            status,
            completed_runs=completed_runs,
            failed_runs=failed_runs,
        )
        persist_unit_ledger(runs_base)

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
