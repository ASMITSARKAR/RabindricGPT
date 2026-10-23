"""
tests/eval/test_humaneval.py : Tests for blind human evaluation sheet randomization and key separation.
"""

import json
import tempfile
from pathlib import Path

from src.rgpt.eval.humaneval import export_blind_sheet


def test_blind_sheet_anonymization() -> None:
    """Human evaluation sheets must have randomized sample IDs without tokenizer names in table."""
    samples = {
        "bpe16": ["লেখা ১", "লেখা ২"],
        "byte": ["লেখা ৩", "লেখা ৪"],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        export_blind_sheet(samples, tmp_dir, seed=42)

        sheet_file = Path(tmp_dir) / "eval_sheet.csv"
        key_file = Path(tmp_dir) / "key_file.json"

        assert sheet_file.exists()
        assert key_file.exists()

        sheet_content = sheet_file.read_text(encoding="utf-8")
        # Ensure tokenizer names are NOT leaked into evaluation sheet
        assert "bpe16" not in sheet_content
        assert "byte" not in sheet_content
        assert "SAMP_" in sheet_content

        with open(key_file, "r", encoding="utf-8") as f:
            key_data = json.load(f)

        assert len(key_data) == 4
        # Key must map sample IDs to tokenizers
        sample_keys = list(key_data.keys())
        assert key_data[sample_keys[0]]["tokenizer"] in ["bpe16", "byte"]
