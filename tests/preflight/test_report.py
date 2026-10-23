"""
tests/preflight/test_report.py : Tests for results/preflight_report.json generation and gate verification.
"""

import tempfile
from pathlib import Path

from src.rgpt.preflight.report import verify_preflight_for_run, write_preflight_report


def test_preflight_report_verification_for_run() -> None:
    """verify_preflight_for_run must return True only when report exists, fingerprint matches, and P6 passed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = Path(tmp_dir) / "preflight_report.json"

        # Report with green levels and run1 approved
        write_preflight_report(
            levels={"P0": "pass", "P1": "pass", "P2": "pass"},
            runs={"run_ok": "pass", "run_fail": "fail"},
            failures=[],
            output_path=report_path,
        )

        assert verify_preflight_for_run("run_ok", report_path=report_path)
        assert not verify_preflight_for_run("run_fail", report_path=report_path)
        assert not verify_preflight_for_run("unknown_run", report_path=report_path)
