"""
tests/data/test_stage1_invariants.py : Tests for audit_stage1_invariants.
"""

import json
import pytest
from pathlib import Path
from scripts.audits.audit_stage1_invariants import audit_stage1_invariants


def test_audit_stage1_invariants_passes_on_valid_data(tmp_path: Path):
    train_file = tmp_path / "train.jsonl"
    egen_file = tmp_path / "egen.jsonl"
    tok_sample_file = tmp_path / "tok_sample.txt"

    egen_file.write_text(
        json.dumps({"id": "egen_1", "text": "ই-জেন সংরক্ষিত বাক্য।"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tok_sample_file.write_text("টোকেনাইজার নমুনার একটি বাক্য।\n", encoding="utf-8")

    # Valid doc containing allowed ZWJ: ক্‍য (U+0995 + U+09CD + U+200D + U+09AF)
    valid_text_1 = "বাংলা সাহিত্যের ইতিহাস অতি প্রাচীন ও সমৃদ্ধ।"
    valid_text_2 = "বাক্যটিতে ক্‍য ফলা ব্যবহৃত হয়েছে সঠিকভাবে।"

    lines = [
        json.dumps({"id": "doc_1", "text": valid_text_1}, ensure_ascii=False),
        json.dumps({"id": "doc_2", "text": valid_text_2}, ensure_ascii=False),
    ]
    train_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = audit_stage1_invariants(
        train_path=train_file,
        egen_path=egen_file,
        tok_sample_path=tok_sample_file,
    )
    assert res["no_duplicates"] is True
    assert res["disjoint_egen"] is True
    assert res["disjoint_tok_sample"] is True
    assert res["invariants_passed"] is True


def test_audit_stage1_invariants_catches_violations(tmp_path: Path):
    train_file = tmp_path / "train.jsonl"
    egen_file = tmp_path / "egen.jsonl"
    tok_sample_file = tmp_path / "tok_sample.txt"

    egen_file.write_text(
        json.dumps({"id": "doc_egen", "text": "কমন টেক্সট।"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tok_sample_file.write_text("নমুনা টেক্সট।\n", encoding="utf-8")

    # Document with BOM and invalid ZWNJ and duplicate ID
    lines = [
        json.dumps({"id": "doc_bad", "text": "\ufeffবম যুক্ত বাক্য।"}, ensure_ascii=False),
        json.dumps({"id": "doc_bad", "text": "জেডডব্লিউএনজে \u200c যুক্ত বাক্য।"}, ensure_ascii=False),
        json.dumps({"id": "doc_egen", "text": "কমন টেক্সট।"}, ensure_ascii=False),
    ]
    train_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    res = audit_stage1_invariants(
        train_path=train_file,
        egen_path=egen_file,
        tok_sample_path=tok_sample_file,
    )
    assert res["duplicate_id_count"] == 1
    assert res["feff_violations"] == 1
    assert res["zwnj_violations"] == 1
    assert res["egen_id_collisions"] == 1
    assert res["egen_hash_collisions"] == 1
