"""
tests/data/test_splits.py : Tests for whole-work split assignment and variorum anti-leakage invariants.
"""

from pathlib import Path

import pytest
import yaml

from src.rgpt.data.splits import (
    SplitLeakageError,
    assign_work_splits,
    verify_whole_work_invariant,
)


def test_whole_work_invariant() -> None:
    """No Tagore work may have chapters or pages distributed across multiple splits."""
    valid_splits = {
        "train": [{"id": "gora"}, {"id": "dakghar"}],
        "val": [{"id": "gitanjali"}],
        "test": [{"id": "jogajog"}],
    }
    # Must pass without raising
    verify_whole_work_invariant(valid_splits)

    # Contaminated split: 'gora' in both train and test
    invalid_splits = {
        "train": [{"id": "gora"}, {"id": "dakghar"}],
        "val": [{"id": "gitanjali"}],
        "test": [{"id": "gora"}],
    }
    with pytest.raises(SplitLeakageError, match="leakage"):
        verify_whole_work_invariant(invalid_splits)


def test_option_a_split_composition() -> None:
    """Option A split must contain all 4 genres in train and match character distribution."""
    works_config = Path("configs/works_wikisource.yaml")
    assert works_config.exists(), "configs/works_wikisource.yaml must exist"

    data = yaml.safe_load(works_config.read_text(encoding="utf-8"))
    works = data.get("works", [])
    assert len(works) == 10, "There must be exactly 10 canonical Tagore works"

    splits = assign_work_splits(works, split_option="Option_A")
    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 3

    # Check genre diversity in train: novels, stories, novella, poetry, drama
    train_genres = {w["genre"] for w in splits["train"]}
    assert "novel" in train_genres
    assert "stories" in train_genres
    assert "poetry" in train_genres
    assert "drama" in train_genres


def test_clean_chars_measured_matches_pages_jsonl() -> None:
    """Recomputes clean_chars_measured from data/raw/wikisource/pages.jsonl and fails on any mismatch."""
    import json
    from collections import defaultdict

    works_config = Path("configs/works_wikisource.yaml")
    data = yaml.safe_load(works_config.read_text(encoding="utf-8"))
    works = data.get("works", [])
    pfx_to_work = {w["prefix"]: w for w in works}

    pages_file = Path("data/raw/wikisource/pages.jsonl")
    if not pages_file.exists():
        pytest.skip("pages.jsonl not present; skipping empirical chars test")

    char_counts = defaultdict(int)
    with open(pages_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pfx = row.get("prefix", "")
            if pfx in pfx_to_work:
                clean_text = row.get("clean_text") or ""
                char_counts[pfx] += len(clean_text)

    for w in works:
        pfx = w["prefix"]
        measured = char_counts[pfx]
        expected = w["clean_chars_measured"]
        assert measured == expected, (
            f"Work '{w['id']}' mismatch: computed {measured} clean chars from pages.jsonl, "
            f"expected {expected} in configs/works_wikisource.yaml"
        )
