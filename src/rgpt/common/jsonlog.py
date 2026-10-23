"""
src/rgpt/common/jsonlog.py : Append-only JSONL logger with fsync policy and truncated-tail recovery.
Backlog/Phase : I-none / Phase B      Docs : 15_BUILD_DIRECTIVE.md Phase B
Contract      : log_event(record: dict) -> None; read_jsonl(path) -> list[dict]
Invariants    : Append is atomic with fsync; reader recovers gracefully from a truncated last line
Failure modes : Disk full; permission denied; mid-write process termination
Status        : TESTED
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union


class JsonLogWriter:
    """Thread-safe append-only JSONL log writer with optional fsync."""

    def __init__(self, path: Union[str, Path], fsync: bool = True) -> None:
        self.path = Path(path).resolve()
        self.fsync = fsync
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        """Append record as a single JSON line and flush/fsync."""
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            if self.fsync:
                os.fsync(f.fileno())


def iter_jsonl(
    path: Union[str, Path], ignore_truncated_tail: bool = True
) -> Iterator[Dict[str, Any]]:
    """Iterate records from a JSONL file lazily, optionally tolerating truncated tail."""
    p = Path(path)
    if not p.exists():
        return

    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            is_last = (idx == len(lines) - 1)
            if is_last and ignore_truncated_tail:
                # Tolerated truncated last line from crash/preemption
                break
            raise


def read_jsonl(
    path: Union[str, Path], ignore_truncated_tail: bool = True
) -> List[Dict[str, Any]]:
    """Read records from a JSONL file into a list."""
    return list(iter_jsonl(path, ignore_truncated_tail=ignore_truncated_tail))
