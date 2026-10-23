"""
tests/preflight/test_fingerprint.py : Tests for code fingerprinting determinism and modification sensitivity.
"""

import tempfile
from pathlib import Path

from src.rgpt.preflight.fingerprint import compute_code_fingerprint


def test_code_fingerprint_deterministic() -> None:
    """Repeated fingerprint calculations on unchanged repository must return identical SHA256."""
    fp1 = compute_code_fingerprint()
    fp2 = compute_code_fingerprint()
    assert fp1 == fp2
    assert fp1.startswith("sha256:")
    assert len(fp1) == 7 + 64


def test_code_fingerprint_changes_on_source_edit() -> None:
    """Modifying a tracked python file in src/ must alter the computed code fingerprint."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        src_dir = tmp_root / "src"
        src_dir.mkdir()

        dummy_file = src_dir / "test.py"
        dummy_file.write_text("x = 1\n", encoding="utf-8")

        fp_orig = compute_code_fingerprint(tmp_root)

        # Edit source file
        dummy_file.write_text("x = 2\n", encoding="utf-8")
        fp_edited = compute_code_fingerprint(tmp_root)

        assert fp_orig != fp_edited
