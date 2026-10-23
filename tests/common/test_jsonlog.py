"""
tests/common/test_jsonlog.py : Tests for append-only JSONL logger and truncated tail tolerance.
"""

from pathlib import Path

import pytest

from src.rgpt.common.jsonlog import JsonLogWriter, read_jsonl


def test_jsonlog_append_and_read(tmp_path: Path) -> None:
    """Records written sequentially must be read back faithfully."""
    log_file = tmp_path / "metrics.jsonl"
    writer = JsonLogWriter(log_file, fsync=True)

    records = [
        {"step": 10, "loss": 4.5, "split": "train"},
        {"step": 20, "loss": 3.8, "split": "train"},
        {"step": 30, "loss": 3.1, "split": "val", "bpc": 2.4},
    ]

    for rec in records:
        writer.write(rec)

    loaded = read_jsonl(log_file)
    assert len(loaded) == 3
    assert loaded == records


def test_jsonlog_truncated_tail_recovery(tmp_path: Path) -> None:
    """Reader must tolerate a partially-written/truncated trailing line without crashing."""
    log_file = tmp_path / "crashed_metrics.jsonl"
    writer = JsonLogWriter(log_file, fsync=False)

    writer.write({"step": 1, "loss": 5.0})
    writer.write({"step": 2, "loss": 4.8})

    # Simulate crash mid-write on step 3 (corrupted JSON string on last line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write('{"step": 3, "loss": 4.2, "tokens": 1234')

    # Default: ignore_truncated_tail=True recovers 2 valid records
    recovered = read_jsonl(log_file, ignore_truncated_tail=True)
    assert len(recovered) == 2
    assert recovered[0]["step"] == 1
    assert recovered[1]["step"] == 2

    # If ignore_truncated_tail=False, must raise json.JSONDecodeError
    with pytest.raises(Exception):
        read_jsonl(log_file, ignore_truncated_tail=False)
