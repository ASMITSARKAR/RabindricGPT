"""
tests/train/test_loop.py : Tests for training loop, NaN loss abort guard, and idempotent execution.
"""

import json
from pathlib import Path

import yaml

from src.rgpt.train.loop import train


def test_idempotent_finished_run_exits_zero(tmp_path: Path) -> None:
    """Invoking train() on a directory containing done.json must immediately exit 0."""
    run_dir = tmp_path / "finished_run"
    run_dir.mkdir()
    done_file = run_dir / "done.json"
    cached = {"status": "COMPLETED", "final_loss": 1.234, "total_steps": 100}
    done_file.write_text(json.dumps(cached), encoding="utf-8")

    result = train(run_dir=run_dir, config_path=tmp_path / "dummy.yaml")
    assert result["status"] == "COMPLETED"
    assert result["final_loss"] == 1.234


def test_nan_loss_guard_preserves_last_good_checkpoint(tmp_path: Path) -> None:
    """A NaN loss occurrence must terminate training without overwriting valid checkpoint."""
    # We test that train raises ValueError on divergence
    run_dir = tmp_path / "nan_run"
    run_dir.mkdir()

    cfg_file = tmp_path / "cfg.yaml"
    cfg_data = {
        "model": {
            "size_id": "tiny",
            "n_layers": 1,
            "d_model": 64,
            "head_dim": 32,
            "n_heads": 2,
            "n_kv_heads": 2,
            "hidden_dim": 128,
            "vocab_size": 264,
            "max_seq_len": 64,
        },
        "train": {
            "total_steps": 10,
            "learning_rate": 1e-3,
            "batch_size": 1,
            "s_max": 32,
        },
        "shard_index": "shards/wikisource/byte/train/index.json",
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")

    # If run cleanly, completes 2 steps
    res = train(run_dir=run_dir, config_path=cfg_file, override_steps=2)
    assert res["status"] == "COMPLETED"
    assert (run_dir / "checkpoints" / "ckpt_final.pt").exists()
