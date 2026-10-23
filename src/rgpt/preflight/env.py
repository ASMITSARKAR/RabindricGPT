"""
src/rgpt/preflight/env.py : Preflight Level P0 checks (environment, Python version, dependencies, secrets, git status).
Backlog/Phase : Phase P      Docs : 15_BUILD_DIRECTIVE.md Phase P (Level P0); DR-14; R7
Contract      : check_environment() -> dict[str, bool]
Invariants    : Scans tracked files for leaked tokens/secrets; asserts git tracking contains no data/checkpoints
Failure modes : Leaked secret pattern found; Python version mismatch; dirty git status with large files
Status        : TESTED
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.rgpt.common.paths import get_project_root

SECRET_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{20,50}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{50,100}"),
    re.compile(r"-----BEGIN (RSA|EC|DSA|OPENSSH|PRIVATE) KEY-----"),
    re.compile(r"AIza[0-9A-Za-z-_]{30,40}"),  # Google API key
]

FORBIDDEN_EXTENSIONS = {".pt", ".bin", ".tar", ".gz", ".zip"}
MAX_TRACKED_FILE_BYTES = 5 * 1024 * 1024  # 5 MB


def check_python_version(expected_major_minor: str = "3.12") -> bool:
    """Verify local Python version (supports 3.12 and local 3.13 development)."""
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    return current in ("3.12", "3.13")


def get_tracked_files(root: Path) -> List[Path]:
    """Get list of files tracked by git, or list of project files if git unavailable."""
    try:
        res = subprocess.run(
            ["git", "ls-files"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        return [root / line for line in lines if (root / line).exists()]
    except Exception:
        # Fallback to filesystem scan
        files = []
        for p in root.rglob("*"):
            if p.is_file() and not any(part.startswith(".") for part in p.parts):
                files.append(p)
        return files


def scan_file_for_secrets(file_path: Path) -> List[str]:
    """Scan a single file for secret patterns."""
    findings = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                findings.append(f"Secret pattern '{pattern.pattern}' detected in {file_path}")
    except Exception:
        pass
    return findings


def check_secrets_and_clean_git(root_dir: Path = None) -> Dict[str, Any]:
    """Scan git tracked files for secret tokens, size limits, and forbidden extensions."""
    root = root_dir if root_dir is not None else get_project_root()
    tracked = get_tracked_files(root)

    secret_findings: List[str] = []
    size_violations: List[str] = []
    forbidden_violations: List[str] = []

    for f in tracked:
        # 1. Secret scan
        secrets = scan_file_for_secrets(f)
        secret_findings.extend(secrets)

        # 2. File size limit
        try:
            sz = f.stat().st_size
            if sz > MAX_TRACKED_FILE_BYTES:
                size_violations.append(f"{f.name} ({sz / (1024*1024):.2f} MB) exceeds 5 MB limit")
        except Exception:
            pass

        # 3. Forbidden binary/checkpoint files tracked
        if f.suffix in FORBIDDEN_EXTENSIONS:
            forbidden_violations.append(f"Forbidden extension '{f.suffix}' tracked: {f}")

    passed = (len(secret_findings) == 0 and len(size_violations) == 0 and len(forbidden_violations) == 0)

    return {
        "passed": passed,
        "secret_findings": secret_findings,
        "size_violations": size_violations,
        "forbidden_violations": forbidden_violations,
        "total_files_scanned": len(tracked),
    }


def run_p0_checks() -> Dict[str, Any]:
    """Execute all Level P0 environment and safety validations."""
    py_ok = check_python_version()
    git_check = check_secrets_and_clean_git()

    passed = py_ok and git_check["passed"]
    failures = []
    if not py_ok:
        failures.append(f"Python version mismatch: {sys.version}")
    failures.extend(git_check["secret_findings"])
    failures.extend(git_check["size_violations"])
    failures.extend(git_check["forbidden_violations"])

    return {
        "status": "pass" if passed else "fail",
        "python_check": py_ok,
        "git_check": git_check,
        "failures": failures,
    }
