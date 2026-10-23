"""
tests/analysis/test_tables.py : Tests for summary report table generation and run ID provenance tracking.
"""

import json
import tempfile
from pathlib import Path

from src.rgpt.analysis.tables import generate_summary_tables


def test_table_run_id_provenance_retention() -> None:
    """Generated Markdown tables must annotate reported figures with originating run IDs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        run_record = {
            "run_id": "PT-bpe16-10m-s0",
            "tokenizer": "bpe16",
            "model_size": "10m",
            "bpc": 1.4523,
            "ci_lower": 1.4312,
            "ci_upper": 1.4734,
            "bpb": 1.0542,
            "ill_formed_unicode_rate": 0.0,
            "in_vocab_word_rate": 0.985,
        }

        with open(tmp_path / "eval_PT-bpe16-10m-s0.json", "w", encoding="utf-8") as f:
            json.dump(run_record, f)

        tables = generate_summary_tables(tmp_path)
        assert "main_summary_table" in tables
        summary_md = tables["main_summary_table"]

        assert "PT-bpe16-10m-s0" in summary_md
        assert "1.4523" in summary_md
        assert "bpe16" in summary_md
