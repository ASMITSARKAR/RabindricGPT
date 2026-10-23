"""
src/rgpt/common/paths.py : Root and path resolution from RGPT_ROOT, enforcing layout and directory bounds.
Backlog/Phase : I-none / Phase B      Docs : 15_BUILD_DIRECTIVE.md Phase B; 03_DATA_PLAN.md §8; 09 §4
Contract      : get_root() -> Path; resolve_path(rel) -> Path; check_folder_limit(dir) -> bool
Invariants    : Never exceeds 10,000 files per folder; respects RGPT_ROOT override
Failure modes : Directory capacity exceeded (>10,000 items); inaccessible mount
Status        : TESTED
"""

import os
from pathlib import Path
from typing import Union

MAX_ITEMS_PER_FOLDER: int = 10000


def get_project_root() -> Path:
    """Resolve the project repository root.

    Priority:
        1. Environment variable RGPT_ROOT if set and valid.
        2. Ancestor directory containing pyproject.toml or README.md.

    Returns:
        Resolved Path to project root.
    """
    env_root = os.environ.get("RGPT_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.exists():
            return p

    # Find relative to this file
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "README.md").exists():
            return parent
    return current.parents[2]


get_repo_root = get_project_root



def resolve_path(relative_or_absolute: Union[str, Path]) -> Path:
    """Resolve path relative to project root if not already absolute.

    Args:
        relative_or_absolute: Relative or absolute path string or Path.

    Returns:
        Resolved absolute Path.
    """
    p = Path(relative_or_absolute)
    if p.is_absolute():
        return p.resolve()
    return (get_project_root() / p).resolve()


def assert_folder_item_limit(directory: Union[str, Path], limit: int = MAX_ITEMS_PER_FOLDER) -> int:
    """Enforce the Google Drive-friendly max item limit per folder (<10,000 items).

    Args:
        directory: Directory path to inspect.
        limit: Maximum allowed entries (default 10,000).

    Returns:
        Exact item count.

    Raises:
        RuntimeError: If entry count exceeds limit.
    """
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        return 0

    count = sum(1 for _ in dir_path.iterdir())
    if count > limit:
        raise RuntimeError(
            f"Folder item limit exceeded for {dir_path}: {count} items found (limit is {limit})."
        )
    return count
