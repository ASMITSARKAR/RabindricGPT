"""
tests/preflight/test_env.py : Tests for preflight Level P0 environment and secret scanning checks.
"""

import tempfile
from pathlib import Path

from src.rgpt.preflight.env import (
    MAX_TRACKED_FILE_BYTES,
    check_python_version,
    scan_file_for_secrets,
)


def test_p0_secret_scanner_detects_planted_token() -> None:
    """Secret scanner mutation test: planted GitHub token pattern must be detected."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write("def dummy():\n    api_token = 'ghp_1234567890abcdefghijklmnopqrstuvwx'\n")
        tmp_path = Path(tmp.name)

    try:
        findings = scan_file_for_secrets(tmp_path)
        assert len(findings) > 0
        assert "ghp_" in findings[0]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_p0_git_tracked_file_size_limit() -> None:
    """Git tracking size limit constant must equal 5 MB per specification."""
    assert MAX_TRACKED_FILE_BYTES == 5 * 1024 * 1024


def test_python_version_support() -> None:
    """Local environment must report supported Python version (3.12 or 3.13)."""
    assert check_python_version()
