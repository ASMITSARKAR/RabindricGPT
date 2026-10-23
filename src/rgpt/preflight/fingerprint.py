"""
src/rgpt/preflight/fingerprint.py : Deterministic code fingerprinting over src, configs, scripts, colab, and requirements.
Backlog/Phase : Phase P      Docs : 15_BUILD_DIRECTIVE.md Phase P; DR-14
Contract      : compute_code_fingerprint(root_dir: Path) -> str
Invariants    : sha256 over sorted file contents; excludes results/ and tests/; stable across platforms
Failure modes : Missing tracked directory; unstaged git modifications
Status        : TESTED
"""

import hashlib
from pathlib import Path
from typing import List, Union

from src.rgpt.common.paths import get_project_root


def compute_code_fingerprint(root_dir: Union[str, Path, None] = None) -> str:
    """Compute deterministic SHA256 fingerprint over tracked source and config files.

    Tracked subtrees:
        - src/
        - configs/
        - scripts/
        - colab/
        - requirements.txt

    Returns:
        Hex-encoded SHA256 string (format: 'sha256:...').
    """
    root = Path(root_dir) if root_dir is not None else get_project_root()
    hasher = hashlib.sha256()

    files_to_hash: List[Path] = []

    # Requirements
    req_file = root / "requirements.txt"
    if req_file.exists():
        files_to_hash.append(req_file)

    # Subdirectories
    subdirs = ["src", "configs", "scripts", "colab"]
    for sd in subdirs:
        dir_path = root / sd
        if dir_path.exists():
            for p in dir_path.rglob("*"):
                if p.is_file() and not p.name.endswith(".pyc") and "__pycache__" not in p.parts:
                    files_to_hash.append(p)

    # Sort deterministically by POSIX relative path
    files_to_hash.sort(key=lambda p: p.relative_to(root).as_posix())

    for f in files_to_hash:
        rel_path = f.relative_to(root).as_posix().encode("utf-8")
        hasher.update(rel_path)
        # Read file bytes (normalize CRLF to LF for cross-platform stability)
        content = f.read_bytes().replace(b"\r\n", b"\n")
        hasher.update(content)

    return f"sha256:{hasher.hexdigest()}"
