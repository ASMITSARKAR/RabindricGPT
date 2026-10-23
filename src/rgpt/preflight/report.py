"""
src/rgpt/preflight/report.py : Preflight report serializer and verifier (results/preflight_report.json).
Backlog/Phase : Phase P      Docs : 15_BUILD_DIRECTIVE.md Phase P; DR-14
Contract      : write_preflight_report(...) -> Path; verify_preflight_for_run(...) -> bool
Invariants    : Atomic write; records code_fingerprint, level statuses, and per-run P6 verifications
Failure modes : Report missing; stale code_fingerprint mismatch; level marked failed
Status        : TESTED
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Union

from src.rgpt.common.atomic_io import atomic_write_json
from src.rgpt.common.paths import get_project_root
from src.rgpt.preflight.fingerprint import compute_code_fingerprint


def write_preflight_report(
    levels: Dict[str, str],
    runs: Dict[str, str],
    failures: List[str],
    output_path: Union[str, Path] = "results/preflight_report.json",
) -> Path:
    """Generate and write results/preflight_report.json with current code fingerprint."""
    root = get_project_root()
    out_file = Path(output_path)
    if not out_file.is_absolute():
        out_file = root / out_file

    current_fingerprint = compute_code_fingerprint(root)
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    report_payload = {
        "code_fingerprint": current_fingerprint,
        "python": py_version,
        "platform": sys.platform,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "levels": levels,
        "runs": runs,
        "failures": failures,
    }

    atomic_write_json(out_file, report_payload)
    return out_file


def verify_preflight_for_run(
    run_id: str,
    report_path: Union[str, Path] = "results/preflight_report.json",
) -> bool:
    """Check if current code fingerprint has a valid green preflight report covering run_id."""
    root = get_project_root()
    p = Path(report_path)
    if not p.is_absolute():
        p = root / p

    if not p.exists():
        return False

    try:
        report = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False

    # 1. Fingerprint freshness check
    current_fp = compute_code_fingerprint(root)
    if report.get("code_fingerprint") != current_fp:
        return False

    # 2. Level statuses: P0, P1, P2 must be "pass"
    levels = report.get("levels", {})
    for req_level in ["P0", "P1", "P2"]:
        if levels.get(req_level) != "pass":
            return False

    # 3. Check specific run ID in runs dictionary
    runs = report.get("runs", {})
    if runs.get(run_id) != "pass":
        return False

    return True
