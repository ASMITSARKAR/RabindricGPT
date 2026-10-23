"""
src/rgpt/common/atomic_io.py : Atomic write-to-temp-then-rename IO operations for checkpoints, indices, and reports.
Backlog/Phase : I-none / Phase B      Docs : 15_BUILD_DIRECTIVE.md Phase B
Contract      : atomic_write(target_path, data) -> None
Invariants    : Target file is never in a partially-written or corrupted state if write is interrupted
Failure modes : Same filesystem temp move failure; permission denied; disk full
Status        : TESTED
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Union


def atomic_write_bytes(target_path: Union[str, Path], data: bytes) -> None:
    """Write data to a temporary file in the target directory and atomically rename.

    Args:
        target_path: Ultimate destination path.
        data: Binary payload.
    """
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=".tmp_", suffix=".tmp", delete=False
    ) as tmp_file:
        tmp_file.write(data)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_name = tmp_file.name

    try:
        os.replace(temp_name, target)
    except Exception:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise


def atomic_write_text(target_path: Union[str, Path], text: str, encoding: str = "utf-8") -> None:
    """Write text atomically to target destination.

    Args:
        target_path: Ultimate destination path.
        text: String content.
        encoding: Text encoding (default UTF-8).
    """
    atomic_write_bytes(target_path, text.encode(encoding))


def atomic_custom_write(target_path: Union[str, Path], writer_fn: Callable[[Path], None]) -> None:
    """Execute a custom writer function against a temporary file, then atomically rename.

    Args:
        target_path: Ultimate destination path.
        writer_fn: Callable taking the temp Path as argument.
    """
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=".tmp_", suffix=".tmp", delete=False
    ) as tmp_file:
        temp_path = Path(tmp_file.name)

    try:
        writer_fn(temp_path)
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_json(target_path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Serialize data to JSON and atomically write to target file."""
    import json
    atomic_write_text(target_path, json.dumps(data, indent=indent, ensure_ascii=False))

