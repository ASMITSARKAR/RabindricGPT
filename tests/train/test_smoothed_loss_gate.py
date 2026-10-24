"""
tests/train/test_smoothed_loss_gate.py : CPU unit test for the 450-500 smoothed-loss relative kill gate.
Ref: 15_BUILD_DIRECTIVE.md Phase I, K; DR-5; Relative kill rule at step 500.
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from src.rgpt.train.loop import train


def test_smoothed_loss_gate_triggers_abort() -> None:
    """If smoothed loss across steps 450-500 has not dropped by required margin, loop must abort."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tdir = Path(tmp_dir)
        cfg_file = tdir / "config.yaml"
        # Configure model with tiny architecture and impossible kill ratio (0.01) to force abort at step 500
        cfg = {
            "run_id": "TEST-GATE-ABORT",
            "stage": "pretrain",
            "tokenizer": "byte",
            "seed": 42,
            "model": {
                "size_id": "tiny",
                "n_layers": 1,
                "d_model": 32,
                "head_dim": 16,
                "n_heads": 2,
                "n_kv_heads": 2,
                "hidden_dim": 64,
                "vocab_size": 264,
                "max_seq_len": 32,
            },
            "train": {
                "total_steps": 505,
                "learning_rate": 0.001,
                "batch_size": 2,
                "s_max": 32,
                "checkpoint_interval": 100,
            },
            "kill_relative_ratio": 0.01,  # Requires 99% drop by step 500, will trigger abort
            "kill_absolute_loss": 0.001,
            "shard_index": "shards/fineweb2/byte/train/index.json",
        }
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

        # Must raise RuntimeError at step 500
        with pytest.raises(RuntimeError) as exc_info:
            train(tdir / "run", cfg_file, override_steps=505)

        assert "Relative kill rule triggered at step 500" in str(exc_info.value)


def test_smoothed_loss_gate_passes_when_healthy() -> None:
    """When smoothed loss is within bounds, training proceeds past step 500 without aborting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tdir = Path(tmp_dir)
        cfg_file = tdir / "config.yaml"
        # Permissive kill ratio (1.5) ensures normal training passes step 500
        cfg = {
            "run_id": "TEST-GATE-PASS",
            "stage": "pretrain",
            "tokenizer": "byte",
            "seed": 42,
            "model": {
                "size_id": "tiny",
                "n_layers": 1,
                "d_model": 32,
                "head_dim": 16,
                "n_heads": 2,
                "n_kv_heads": 2,
                "hidden_dim": 64,
                "vocab_size": 264,
                "max_seq_len": 32,
            },
            "train": {
                "total_steps": 502,
                "learning_rate": 0.001,
                "batch_size": 2,
                "s_max": 32,
                "checkpoint_interval": 200,
            },
            "kill_relative_ratio": 1.5,
            "kill_absolute_loss": 50.0,
            "shard_index": "shards/fineweb2/byte/train/index.json",
        }
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

        res = train(tdir / "run", cfg_file, override_steps=502)
        assert res["status"] == "COMPLETED"
        assert res["total_steps"] == 502
