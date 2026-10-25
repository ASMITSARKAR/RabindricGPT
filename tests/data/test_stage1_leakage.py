"""
tests/data/test_stage1_leakage.py : Tests for audit_stage1_leakage.
"""

import pytest
from pathlib import Path
from scripts.audits.audit_stage1_leakage import audit_stage1_leakage


def test_stage1_leakage_plant_recall_passes_threshold():
    """Plant 20 verbatim 13-grams per work into synthetic docs and assert >= 95% recall."""
    if not Path("data/raw/wikisource/pages.jsonl").exists():
        pytest.skip("data/raw/wikisource/pages.jsonl absent in clean clone; run ingest first.")
    train_path = Path("data/raw/fineweb2/train.jsonl")
    res = audit_stage1_leakage(train_path=train_path, n_plants_per_work=20, seed=42)

    assert res["all_passed"] is True, f"Recall test failed: {res['recall_results']}"
    assert res["total_recall"] >= 95.0, f"Total recall was {res['total_recall']}%"
    for w_key, r in res["recall_results"].items():
        assert r["recall_pct"] >= 95.0, f"Work {w_key} recall below 95%: {r['recall_pct']}%"
