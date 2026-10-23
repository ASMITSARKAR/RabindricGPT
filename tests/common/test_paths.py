"""
tests/common/test_paths.py : Tests for repository root resolution and 10,000 folder entry bound.
"""

from pathlib import Path

import pytest

from src.rgpt.common.paths import (
    assert_folder_item_limit,
    get_project_root,
    resolve_path,
)


def test_project_root_resolution() -> None:
    """Project root must locate RabindricGPT workspace directory correctly."""
    root = get_project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "configs" / "decisions.yaml").exists()


def test_resolve_path_relative_and_absolute() -> None:
    """resolve_path must handle both relative and absolute paths accurately."""
    root = get_project_root()
    rel = resolve_path("configs/data.yaml")
    assert rel == root / "configs" / "data.yaml"

    abs_p = (root / "README.md").resolve()
    assert resolve_path(abs_p) == abs_p


def test_folder_entry_bound_enforcement(tmp_path: Path) -> None:
    """Directories exceeding folder limit must raise RuntimeError."""
    test_dir = tmp_path / "bounded_dir"
    test_dir.mkdir()

    for i in range(15):
        (test_dir / f"file_{i}.txt").touch()

    # Under limit: passes
    assert assert_folder_item_limit(test_dir, limit=20) == 15

    # Over limit: raises
    with pytest.raises(RuntimeError, match="limit exceeded"):
        assert_folder_item_limit(test_dir, limit=10)
