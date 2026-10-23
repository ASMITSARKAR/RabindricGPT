"""
tests/compute/test_ledger.py : Tests for compute ledger row summation and report generation.
"""

import tempfile
from pathlib import Path

import pytest

from src.rgpt.common.atomic_io import atomic_write_json
from src.rgpt.compute.ledger import update_ledger_report


def test_ledger_sum_matches_row_totals() -> None:
    """Aggregated ledger report must exactly equal the mathematical sum of individual run rows."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        # Create 3 run directories with ledger rows
        rows = [
            {"run_id": "PT-bpe16-10m-s0", "stage": "pretrain_10m", "tokenizer": "bpe16", "model_size": "10m", "spent_units": 0.47, "gpu": "L4"},
            {"run_id": "PT-byte-10m-s0", "stage": "pretrain_10m", "tokenizer": "byte", "model_size": "10m", "spent_units": 7.20, "gpu": "L4"},
            {"run_id": "PT-char-10m-s0", "stage": "pretrain_10m", "tokenizer": "char", "model_size": "10m", "spent_units": 1.60, "gpu": "L4"},
        ]
        for r in rows:
            r_dir = runs_dir / r["run_id"]
            r_dir.mkdir()
            atomic_write_json(r_dir / "ledger_row.json", r)

        out_md = tmp_path / "ledger.md"
        rep = update_ledger_report(runs_dir, out_md)

        expected_total = 0.47 + 7.20 + 1.60
        assert pytest.approx(rep["total_spent_units"], abs=1e-3) == expected_total
        assert rep["num_runs"] == 3
        assert out_md.exists()

        content = out_md.read_text(encoding="utf-8")
        assert "PT-bpe16-10m-s0" in content
        assert "PT-byte-10m-s0" in content
