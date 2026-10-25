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


def test_runner_stops_at_budget_ceiling_and_writes_final_summary() -> None:
    """Runner must stop when cumulative spend reaches ceiling (165 u) and write final_summary.json."""
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        runs_base = Path(tmp) / "runs"
        runs_base.mkdir(parents=True, exist_ok=True)

        q_path = Path(tmp) / "queue.yaml"
        queue_content = {
            "runs": [
                {"run_id": "RUN_A", "approved": True, "smoke": True, "enforce_gate": False},
                {"run_id": "RUN_B", "approved": True, "smoke": True, "enforce_gate": False},
                {"run_id": "RUN_C", "approved": True, "smoke": True, "enforce_gate": False},
            ]
        }
        with open(q_path, "w", encoding="utf-8") as f:
            yaml.dump(queue_content, f)

        executed = []
        def mock_runner(rid: str) -> None:
            executed.append(rid)
            rdir = runs_base / rid
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "done.json").write_text(json.dumps({
                "status": "COMPLETED",
                "tokens_per_sec": 51200.0,
                "final_loss": 5.42,
            }), encoding="utf-8")
            # Spend 100u on RUN_A, 70u on RUN_B -> sum = 170u > 165u ceiling
            spent = 100.0 if rid == "RUN_A" else 70.0
            (rdir / "ledger_row.json").write_text(json.dumps({
                "run_id": rid,
                "spent_units": spent,
                "tokens_per_sec": 51200.0,
                "time_seconds": 120.0,
            }), encoding="utf-8")

        completed = run_supervised_queue(
            queue_path=str(q_path),
            disconnect_hook=lambda: None,
            runner_fn=mock_runner,
            runs_base=runs_base,
            budget_ceiling=165.0,
        )

        # RUN_A and RUN_B completed, RUN_C must NOT be executed because ceiling (165u) was exceeded
        assert executed == ["RUN_A", "RUN_B"]
        assert completed == ["RUN_A", "RUN_B"]
        assert "RUN_C" not in completed

        # Verify final_summary.json exists and has expected fields
        final_summary_file = runs_base / "final_summary.json"
        assert final_summary_file.exists(), "final_summary.json was not written"
        summary_data = json.loads(final_summary_file.read_text(encoding="utf-8"))
        assert summary_data["status"] == "STOPPED_AT_CEILING"
        assert summary_data["stopped_by_ceiling"] is True
        assert summary_data["total_spent_units"] == 170.0
        assert summary_data["ceiling_units"] == 165.0
        assert summary_data["completed_count"] == 2
        assert len(summary_data["runs"]) == 2


def test_runner_no_retry_on_oom() -> None:
    """Runner must not retry on CUDA out of memory errors (immediate hard stop)."""
    attempts = []

    def oom_runner(rid: str) -> None:
        attempts.append(len(attempts) + 1)
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(RunnerError) as exc:
        execute_run("test_oom_run", max_attempts=5, runner_fn=oom_runner)

    assert len(attempts) == 1, "Runner should not retry on OOM error!"
    assert "fatal non-retriable error" in str(exc.value)


def test_runner_no_retry_on_gate_abort() -> None:
    """Runner must not retry on queue gate abort."""
    attempts = []

    def gate_fail_runner(rid: str) -> None:
        attempts.append(len(attempts) + 1)
        raise QueueGateError("Queue gate violation: smoothed loss >= 7.5")

    with pytest.raises(RunnerError) as exc:
        execute_run("test_gate_fail_run", max_attempts=5, runner_fn=gate_fail_runner)

    assert len(attempts) == 1, "Runner should not retry on gate abort!"
    assert "fatal non-retriable error" in str(exc.value)


def test_runner_resume_from_checkpoint() -> None:
    """Runner must identify existing checkpoints and support resumption."""
    import tempfile
    import torch
    from src.rgpt.train.checkpoint import load_latest_checkpoint, save_checkpoint
    from src.rgpt.model.gpt import RabindricGPT
    from src.rgpt.model.config import ModelConfig

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "resume_run"
        run_dir.mkdir(parents=True)

        m = RabindricGPT(ModelConfig(
            size_id="tiny", n_layers=1, d_model=64, head_dim=32, n_heads=2, n_kv_heads=2, hidden_dim=128, vocab_size=256, max_seq_len=32
        ))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        ckpt_saved = save_checkpoint(
            checkpoint_dir=run_dir,
            step=100,
            model=m,
            optimizer=opt,
            scaler=None,
            config_hash="test_cfg",
            tokenizer_hash="bpe16",
        )
        assert ckpt_saved.exists()

        # Load latest checkpoint
        ckpt_loaded = load_latest_checkpoint(run_dir, expected_config_hash="test_cfg")
        assert ckpt_loaded is not None
        assert ckpt_loaded["step"] == 100
        assert ckpt_loaded["config_hash"] == "test_cfg"


def test_ledger_not_double_counted() -> None:
    """Multiple calls to persist_unit_ledger must not double-count spent units."""
    import json
    import tempfile
    from colab.runner_lib import persist_unit_ledger

    with tempfile.TemporaryDirectory() as tmp:
        runs_base = Path(tmp) / "runs"
        runs_base.mkdir(parents=True)

        # Create two runs with ledger rows
        for rid, spent in [("RUN_1", 12.5), ("RUN_2", 8.2)]:
            rdir = runs_base / rid
            rdir.mkdir(parents=True)
            (rdir / "ledger_row.json").write_text(json.dumps({
                "run_id": rid,
                "spent_units": spent,
                "gpu": "L4",
                "time_seconds": 100.0,
            }), encoding="utf-8")

        # First ledger aggregation
        rep1 = persist_unit_ledger(runs_base)
        assert rep1["total_spent_units"] == 20.7
        assert rep1["num_runs"] == 2

        # Second ledger aggregation (must be identical, no double-counting)
        rep2 = persist_unit_ledger(runs_base)
        assert rep2["total_spent_units"] == 20.7
        assert rep2["num_runs"] == 2

        # Third ledger aggregation
        rep3 = persist_unit_ledger(runs_base)
        assert rep3["total_spent_units"] == 20.7
        assert rep3["num_runs"] == 2


def test_runner_unassign_when_done() -> None:
    """Runner must call disconnect_hook (runtime.unassign) when done."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        q_path = Path(tmp) / "queue.yaml"
        queue_content = {"runs": []}
        with open(q_path, "w", encoding="utf-8") as f:
            yaml.dump(queue_content, f)

        unassigned = []
        completed = run_supervised_queue(
            queue_path=str(q_path),
            disconnect_hook=lambda: unassigned.append(True),
            runs_base=Path(tmp) / "runs",
        )
        assert len(unassigned) == 1, "disconnect_hook was not called when queue finished!"


def test_bootstrap_notebook_one_idempotent_cell_and_paths_exist() -> None:
    """colab/bootstrap.ipynb must parse as valid JSON, have exactly one code cell, and referenced paths must exist."""
    import json
    from src.rgpt.common.paths import get_project_root

    root = get_project_root()
    nb_path = root / "colab" / "bootstrap.ipynb"
    assert nb_path.exists(), f"Notebook missing at {nb_path}"

    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb_data.get("cells", [])
    assert len(cells) == 1, f"bootstrap.ipynb must have exactly one cell, found {len(cells)}"
    assert cells[0].get("cell_type") == "code", "bootstrap.ipynb cell must be a code cell"

    cell_src = "".join(cells[0].get("source", []))
    assert "!git clone" in cell_src
    assert "!git -C /content/RabindricGPT pull --ff-only" in cell_src
    assert "drive_runs = drive_base / 'runs'" in cell_src
    assert "os.symlink(str(drive_runs), str(local_runs))" in cell_src

    # Check that referenced repository files exist
    assert (root / "configs" / "runs" / "queue.yaml").exists()
    assert (root / "scripts" / "00_preflight.py").exists()
    assert (root / "colab" / "runner_lib.py").exists()


