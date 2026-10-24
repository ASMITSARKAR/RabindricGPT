"""
tests/colab/test_runner_lib.py : Tests for Colab supervised runner logic, retry supervisor, and preflight gates.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from colab.runner_lib import (
    ConfigHashMismatchError,
    QueueGateError,
    RunnerError,
    execute_run,
    run_supervised_queue,
    verify_preflight_and_fingerprint,
    verify_queue_gate,
)


def test_runner_refuses_unapproved_runs() -> None:
    """Supervised runner must never launch a queue entry where approved is false."""
    executed = []

    def mock_runner(rid: str) -> None:
        executed.append(rid)

    with tempfile.TemporaryDirectory() as tmp_dir:
        q_path = Path(tmp_dir) / "queue.yaml"
        queue_content = {
            "queue": [
                {"run_id": "RUN_APPROVED", "approved": True, "smoke": True},
                {"run_id": "RUN_UNAPPROVED", "approved": False, "smoke": True},
            ]
        }
        with open(q_path, "w", encoding="utf-8") as f:
            yaml.dump(queue_content, f)

        disconnected = []
        completed = run_supervised_queue(
            queue_path=str(q_path),
            disconnect_hook=lambda: disconnected.append(True),
            runner_fn=mock_runner,
        )

        assert "RUN_APPROVED" in executed
        assert "RUN_UNAPPROVED" not in executed
        assert "RUN_APPROVED" in completed
        assert len(disconnected) == 1


def test_runner_retry_and_attempt_cap() -> None:
    """Runner must retry on failure up to max_attempts then cleanly disconnect runtime."""
    attempts = []

    def failing_runner(rid: str) -> None:
        attempts.append(len(attempts) + 1)
        raise RuntimeError("Simulated failure")

    with pytest.raises(RunnerError):
        execute_run("test_fail_run", max_attempts=3, runner_fn=failing_runner)

    assert len(attempts) == 3


def test_runner_config_hash_mismatch_hard_stop() -> None:
    """If run config hash mismatches checkpoint, runner must hard-stop without retrying."""
    attempts = []

    def mismatch_runner(rid: str) -> None:
        attempts.append(len(attempts) + 1)
        raise ConfigHashMismatchError("Config hash mismatch detected")

    with pytest.raises(ConfigHashMismatchError):
        execute_run("test_mismatch_run", max_attempts=5, runner_fn=mismatch_runner)

    # Must hard-stop on attempt 1, without retrying
    assert len(attempts) == 1


def test_runner_refuses_on_missing_preflight() -> None:
    """verify_preflight_and_fingerprint must refuse non-smoke runs if preflight report is missing."""
    with tempfile.TemporaryDirectory() as tmp:
        # With non-existent report in temporary directory
        with pytest.raises(RunnerError):
            verify_preflight_and_fingerprint("PT-bpe16-10m-s0", is_smoke=False, root_dir=Path(tmp))


def test_runner_skips_done_runs() -> None:
    """Runner must skip runs that already have done.json."""
    with tempfile.TemporaryDirectory() as tmp:
        runs_base = Path(tmp) / "runs"
        done_run_dir = runs_base / "DONE_RUN"
        done_run_dir.mkdir(parents=True, exist_ok=True)
        (done_run_dir / "done.json").write_text('{"status": "COMPLETED"}', encoding="utf-8")

        called = []
        def mock_runner(rid: str) -> None:
            called.append(rid)

        # Direct execute_run must return True and not invoke runner_fn
        res = execute_run("DONE_RUN", max_attempts=3, runner_fn=mock_runner, runs_base=runs_base)
        assert res is True
        assert len(called) == 0


def test_runner_status_json_persistence() -> None:
    """Runner must write status.json tracking current run and completion status."""
    import json
    with tempfile.TemporaryDirectory() as tmp:
        runs_base = Path(tmp) / "runs"
        runs_base.mkdir(parents=True, exist_ok=True)

        q_path = Path(tmp) / "queue.yaml"
        queue_content = {
            "queue": [
                {"run_id": "TEST_RUN_1", "approved": True, "smoke": True},
            ]
        }
        with open(q_path, "w", encoding="utf-8") as f:
            yaml.dump(queue_content, f)

        def mock_runner(rid: str) -> None:
            rdir = runs_base / rid
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "done.json").write_text('{"status": "COMPLETED"}', encoding="utf-8")

        completed = run_supervised_queue(
            queue_path=str(q_path),
            disconnect_hook=lambda: None,
            runner_fn=mock_runner,
            runs_base=runs_base,
        )

        assert "TEST_RUN_1" in completed
        status_file = runs_base / "status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["status"] == "COMPLETED"
        assert "TEST_RUN_1" in status_data["completed_runs"]


def test_queue_gate_requires_bpe16_first() -> None:
    """Queue gate must refuse non-bpe16 run if PT-bpe16-10m-s0 has not completed."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        runs_base = Path(tmp)
        with pytest.raises(QueueGateError) as exc:
            verify_queue_gate("PT-akshara-10m-s0", runs_base=runs_base)
        assert "PT-bpe16-10m-s0 must run and complete first" in str(exc.value)


def test_queue_gate_aborts_when_bpe16_smoothed_loss_exceeds_threshold() -> None:
    """Queue gate must halt and refuse subsequent runs if PT-bpe16-10m-s0 smoothed loss >= 7.5."""
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        runs_base = Path(tmp)
        bpe16_dir = runs_base / "PT-bpe16-10m-s0"
        bpe16_dir.mkdir(parents=True)
        # Record bad smoothed loss >= 7.5
        (bpe16_dir / "done.json").write_text(json.dumps({
            "status": "COMPLETED",
            "smoothed_loss_500": 7.82,
        }), encoding="utf-8")

        with pytest.raises(QueueGateError) as exc:
            verify_queue_gate("PT-akshara-10m-s0", runs_base=runs_base)
        assert "smoothed loss (7.8200) >= 7.5 ceiling" in str(exc.value)


def test_queue_gate_allows_subsequent_runs_when_bpe16_passes() -> None:
    """Queue gate must permit subsequent runs when PT-bpe16-10m-s0 smoothed loss < 7.5."""
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        runs_base = Path(tmp)
        bpe16_dir = runs_base / "PT-bpe16-10m-s0"
        bpe16_dir.mkdir(parents=True)
        # Record good smoothed loss < 7.5 (e.g. 6.42)
        (bpe16_dir / "done.json").write_text(json.dumps({
            "status": "COMPLETED",
            "smoothed_loss_500": 6.42,
        }), encoding="utf-8")

        # Must not raise
        verify_queue_gate("PT-akshara-10m-s0", runs_base=runs_base)

