"""
tests/common/test_atomic_io.py : Tests for atomic file write operations under simulated crashes.
"""

from pathlib import Path

import pytest

from src.rgpt.common.atomic_io import (
    atomic_custom_write,
    atomic_write_bytes,
    atomic_write_text,
)


def test_atomic_write_bytes(tmp_path: Path) -> None:
    """Atomic write must produce valid destination file without intermediate state."""
    target = tmp_path / "data.bin"
    payload = b"Hello, RabindricGPT atomic io!"
    atomic_write_bytes(target, payload)

    assert target.exists()
    assert target.read_bytes() == payload


def test_atomic_write_text(tmp_path: Path) -> None:
    """Atomic text write must handle UTF-8 properly."""
    target = tmp_path / "poem.txt"
    bengali_stanza = "চিত্ত যেথা ভয়শূন্য, উচ্চ যেথা শির"
    atomic_write_text(target, bengali_stanza)

    assert target.exists()
    assert target.read_text(encoding="utf-8") == bengali_stanza


def test_atomic_write_crash_simulation(tmp_path: Path) -> None:
    """If write process terminates mid-write, original destination file must remain untouched."""
    target = tmp_path / "original.txt"
    original_content = "pristine original state"
    atomic_write_text(target, original_content)

    def failing_writer(temp_path: Path) -> None:
        temp_path.write_text("partial corrupt state")
        raise RuntimeError("Simulated crash mid-write")

    with pytest.raises(RuntimeError, match="Simulated crash"):
        atomic_custom_write(target, failing_writer)

    # Original file must remain 100% intact
    assert target.read_text() == original_content

    # No leftover temporary files in directory
    temp_files = list(tmp_path.glob(".tmp_*"))
    assert len(temp_files) == 0
