"""
tests/data/test_ingest_leakage_invariants.py : Strict fail-loud invariant tests and
post-hoc deduplication & leakage cleaning tests for Stage 1 data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import importlib

ingest_mod = importlib.import_module("scripts.02_ingest_fineweb2")
HELD_OUT_WORKS = ingest_mod.HELD_OUT_WORKS
build_per_work_detectors = ingest_mod.build_per_work_detectors
load_reserved_identifiers = ingest_mod.load_reserved_identifiers
run_post_hoc_clean = ingest_mod.run_post_hoc_clean

from scripts.audits.audit_stage1_leakage import audit_stage1_leakage


def _make_dummy_pages_jsonl(pages_path: Path) -> None:
    """Create a synthetic pages.jsonl with 200 distinct Bengali words per held-out work."""
    bengali_digits = "০১২৩৪৫৬৭৮৯"

    def to_bengali_num(n: int) -> str:
        return "".join(bengali_digits[int(d)] for d in str(n))

    work_roots = {
        "shesher_kobita": "লাবণ্যকথা",
        "gitanjali": "অঞ্জলিবাক্য",
        "jogajog": "কুমুদিনীবিবরণ",
        "chitrangada": "মণিপুরবিবরণ",
    }
    lines = []
    for w_key, title in HELD_OUT_WORKS.items():
        root = work_roots[w_key]
        words = [f"{root}{to_bengali_num(i)}" for i in range(200)]
        text = " ".join(words) + " সমাপ্ত।"
        row = {
            "title": f"{w_key} প্রথম সর্গ",
            "prefix": f"{title}/অধ্যায় ১",
            "clean_text": text,
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    pages_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_dummy_egen_jsonl(egen_path: Path) -> None:
    """Create a synthetic egen.jsonl with doc IDs and text."""
    rows = [
        {"id": "egen_doc_001", "text": "এটি কৃত্রিমভাবে প্রস্তুত একটি বাংলা বাক্য যা সংরক্ষিত।"},
        {"id": "egen_doc_002", "text": "দ্বিতীয় সংরক্ষিত ই-জেন বাক্য পরীক্ষার জন্য ব্যবহৃত হইতেছে।"},
    ]
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    egen_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Fail-loud tests for load_reserved_identifiers
# ---------------------------------------------------------------------------

def test_load_reserved_identifiers_missing_file_raises(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nonexistent_egen.jsonl"
    with pytest.raises(FileNotFoundError, match="Missing required file"):
        load_reserved_identifiers(egen_path=nonexistent)


def test_load_reserved_identifiers_empty_file_raises(tmp_path: Path) -> None:
    empty_egen = tmp_path / "empty_egen.jsonl"
    empty_egen.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Reserved-ID set.*empty"):
        load_reserved_identifiers(egen_path=empty_egen)


def test_load_reserved_identifiers_missing_tok_sample_raises(tmp_path: Path) -> None:
    egen_p = tmp_path / "egen.jsonl"
    _make_dummy_egen_jsonl(egen_p)
    with pytest.raises(FileNotFoundError, match="Missing required file"):
        load_reserved_identifiers(egen_path=egen_p, tok_sample_path=tmp_path / "missing_sample.txt")


def test_load_reserved_identifiers_empty_hashes_raises(tmp_path: Path) -> None:
    no_text_egen = tmp_path / "no_text_egen.jsonl"
    no_text_egen.write_text(json.dumps({"id": "only_id_no_text"}) + "\n", encoding="utf-8")
    empty_sample = tmp_path / "empty_sample.txt"
    empty_sample.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Reserved content hash set.*empty"):
        load_reserved_identifiers(egen_path=no_text_egen, tok_sample_path=empty_sample)


# ---------------------------------------------------------------------------
# 2. Fail-loud tests for build_per_work_detectors
# ---------------------------------------------------------------------------

def test_build_per_work_detectors_missing_file_raises(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nonexistent_pages.jsonl"
    with pytest.raises(FileNotFoundError, match="Missing required file"):
        build_per_work_detectors(pages_path=nonexistent)


def test_build_per_work_detectors_missing_work_raises(tmp_path: Path) -> None:
    partial_pages = tmp_path / "partial_pages.jsonl"
    # Only write Shesher Kobita, missing the other 3
    row = {
        "title": "শেষের কবিতা",
        "prefix": "শেষের কবিতা/১",
        "clean_text": "আজকের দিনে অমল ও লাবণ্য দেখা করিল নদীর ধারে বসিয়া কবিতা লিখিল।"
    }
    partial_pages.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No text found for held-out work"):
        build_per_work_detectors(pages_path=partial_pages)


def test_build_per_work_detectors_zero_5grams_raises(tmp_path: Path) -> None:
    short_pages = tmp_path / "short_pages.jsonl"
    # Each work has fewer than 5 words, producing 0 5-grams
    lines = []
    for w_key, title in HELD_OUT_WORKS.items():
        row = {
            "title": title,
            "prefix": f"{title}/১",
            "clean_text": "এক দুই তিন",  # Only 3 words -> 0 5-grams
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    short_pages.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has 0 5-grams indexed"):
        build_per_work_detectors(pages_path=short_pages)


# ---------------------------------------------------------------------------
# 3. Fail-loud tests for audit_stage1_leakage
# ---------------------------------------------------------------------------

def test_audit_stage1_leakage_missing_pages_raises(tmp_path: Path) -> None:
    egen_p = tmp_path / "egen.jsonl"
    _make_dummy_egen_jsonl(egen_p)
    with pytest.raises(FileNotFoundError, match="Missing required file"):
        audit_stage1_leakage(
            pages_path=tmp_path / "missing_pages.jsonl",
            egen_path=egen_p,
        )


def test_audit_stage1_leakage_missing_egen_raises(tmp_path: Path) -> None:
    pages_p = tmp_path / "pages.jsonl"
    _make_dummy_pages_jsonl(pages_p)
    with pytest.raises(FileNotFoundError, match="Missing required file"):
        audit_stage1_leakage(
            pages_path=pages_p,
            egen_path=tmp_path / "missing_egen.jsonl",
        )


def test_audit_stage1_leakage_missing_tok_sample_raises(tmp_path: Path) -> None:
    pages_p = tmp_path / "pages.jsonl"
    egen_p = tmp_path / "egen.jsonl"
    _make_dummy_pages_jsonl(pages_p)
    _make_dummy_egen_jsonl(egen_p)
    with pytest.raises(FileNotFoundError, match="Missing required file"):
        audit_stage1_leakage(
            pages_path=pages_p,
            egen_path=egen_p,
            tok_sample_path=tmp_path / "missing_sample.txt",
        )


# ---------------------------------------------------------------------------
# 4. Post-Hoc Cleaning Mode Tests
# ---------------------------------------------------------------------------

def test_run_post_hoc_clean_filters_all_invalids(tmp_path: Path) -> None:
    pages_p = tmp_path / "pages.jsonl"
    egen_p = tmp_path / "egen.jsonl"
    _make_dummy_pages_jsonl(pages_p)
    _make_dummy_egen_jsonl(egen_p)

    # Read Shesher Kobita text to extract a real leaking 5-gram
    with open(pages_p, "r", encoding="utf-8") as f:
        sk_row = json.loads(f.readline())
    sk_words = sk_row["clean_text"].split()
    leaking_passage = " ".join(sk_words[:10])

    input_train = tmp_path / "raw_train.jsonl"
    output_train = tmp_path / "cleaned_train.jsonl"

    clean_doc_1 = "বাংলা সাহিত্যের ইতিহাস অতি প্রাচীন ও সমৃদ্ধ। প্রাচীন যুগ হইতে আধুনিক যুগ পর্যন্ত বহু সাধক কবি এখানে সাহিত্য চর্চা করিয়াছেন।"
    clean_doc_2 = "শরতের নীল আকাশে সাদা মেঘের ভেলা ভাসিয়া বেড়ায়। নদীর তীরে কাশফুল ফুটিয়া ওঠে এবং গ্রামবাংলার রূপ অপরূপ দেখায়।"

    train_rows = [
        # 1. Clean Doc A
        {"id": "doc_clean_001", "text": clean_doc_1},
        # 2. Duplicate Doc (same ID as Doc A)
        {"id": "doc_clean_001", "text": clean_doc_1},
        # 3. Duplicate Doc (different ID, but identical content/hash as Doc A)
        {"id": "doc_clean_001_copy", "text": clean_doc_1},
        # 4. Reserved Collision by ID (egen_doc_001)
        {"id": "egen_doc_001", "text": "অন্য যেকোনো বাংলা বাক্য যা আইডি মিলে যায়।"},
        # 5. Reserved Collision by Content Hash (egen text)
        {"id": "doc_egen_hash_leak", "text": "এটি কৃত্রিমভাবে প্রস্তুত একটি বাংলা বাক্য যা সংরক্ষিত।"},
        # 6. Narrow Tagore Keyword Leak ("গীতাঞ্জলি")
        {"id": "doc_keyword_leak", "text": "আজকের আলোচনায় রবীন্দ্রনাথ ঠাকুরের বিখ্যাত নোবেলজয়ী গীতাঞ্জলি কাব্যগ্রন্থের পর্যালোচনা করা হইবে।"},
        # 7. Four-Work 5-gram Leak (from Shesher Kobita: >= 3 distinct 5-grams in 30-word span)
        {"id": "doc_ngram_leak", "text": f"অতীতের পাতায় দেখা যায় {leaking_passage} বলিয়া মানুষ মনে রাখিয়াছিল এবং আলোচনা করিয়াছিল।"},
        # 8. Quality Rejection (<80% Bengali script)
        {"id": "doc_quality_reject", "text": "This is completely in English language without any Bengali characters."},
        # 9. Clean Doc B
        {"id": "doc_clean_002", "text": clean_doc_2},
    ]

    with open(input_train, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dummy_tok = tmp_path / "dummy_tok.txt"
    dummy_tok.write_text("নমুনা টোকেনাইজার বাক্য।\n\n", encoding="utf-8")

    report = run_post_hoc_clean(
        input_file=input_train,
        output_file=output_train,
        pages_path=pages_p,
        egen_path=egen_p,
        tok_sample_path=dummy_tok,
    )

    assert report["scanned_docs"] == 9
    assert report["kept_docs"] == 2
    assert report["total_purged"] == 7
    assert report["purged_counts"]["duplicates"] == 2
    assert report["purged_counts"]["reserved_collisions"] == 2
    assert report["purged_counts"]["narrow_keywords"] == 1
    assert report["purged_counts"]["shesher_kobita"] == 1
    assert report["purged_counts"]["quality_rejected"] == 1

    # Verify cleaned file content
    assert output_train.exists()
    cleaned_rows = [json.loads(line) for line in output_train.read_text(encoding="utf-8").strip().split("\n")]
    assert len(cleaned_rows) == 2
    assert cleaned_rows[0]["id"] == "doc_clean_001"
    assert cleaned_rows[1]["id"] == "doc_clean_002"
    assert "bytes" in cleaned_rows[0]
    assert "chars" in cleaned_rows[0]


def test_cli_post_hoc_clean_subprocess(tmp_path: Path) -> None:
    """Verify scripts/02_ingest_fineweb2.py --post-hoc-clean works from CLI."""
    pages_p = tmp_path / "pages.jsonl"
    egen_p = tmp_path / "egen.jsonl"
    _make_dummy_pages_jsonl(pages_p)
    _make_dummy_egen_jsonl(egen_p)

    input_train = tmp_path / "in.jsonl"
    output_train = tmp_path / "out.jsonl"

    clean_text = "বাংলাদেশের প্রাকৃতিক সৌন্দর্য অত্যন্ত মনোরম। সবুজ শ্যামল প্রান্তর ও নদ-নদী এই দেশের প্রধান বৈশিষ্ট্য।"
    rows = [
        {"id": "c1", "text": clean_text},
        {"id": "c1", "text": clean_text},  # duplicate
    ]
    with open(input_train, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dummy_tok = tmp_path / "dummy_tok.txt"
    dummy_tok.write_text("নমুনা টোকেনাইজার বাক্য।\n\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/02_ingest_fineweb2.py",
        "--post-hoc-clean",
        "--input-file", str(input_train),
        "--output-file", str(output_train),
        "--pages-path", str(pages_p),
        "--egen-path", str(egen_p),
        "--tok-sample-path", str(dummy_tok),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    assert "POST-HOC DATA CLEANING AND PURGING MODE" in proc.stdout
    assert "Scanned Documents    : 2" in proc.stdout
    assert "Kept Clean Documents : 1" in proc.stdout
    assert output_train.exists()


def test_cli_fails_loudly_on_missing_pages_subprocess(tmp_path: Path) -> None:
    """Verify CLI exits with non-zero when required reference files are missing."""
    cmd = [
        sys.executable,
        "scripts/02_ingest_fineweb2.py",
        "--pages-path", str(tmp_path / "missing.jsonl"),
        "--egen-path", str(tmp_path / "missing_egen.jsonl"),
        "--chunk-bytes", "1000",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode != 0
    assert "FileNotFoundError" in proc.stderr or "FileNotFoundError" in proc.stdout


def test_clean_stage1_posthoc_script_runs_cleanly(tmp_path: Path) -> None:
    from scripts.clean_stage1_posthoc import clean_train_file

    pages_p = tmp_path / "pages.jsonl"
    egen_p = tmp_path / "egen.jsonl"
    tok_sample_p = tmp_path / "tok_sample.txt"
    _make_dummy_pages_jsonl(pages_p)
    _make_dummy_egen_jsonl(egen_p)
    tok_sample_p.write_text("নমুনা টোকেনাইজার পাঠ্য বাক্য।\n\n", encoding="utf-8")

    input_train = tmp_path / "raw_train.jsonl"
    output_train = tmp_path / "cleaned_train.jsonl"

    clean_text = "বাংলাদেশের ঐতিহাসিক ঐতিহ্য অত্যন্ত সমৃদ্ধ ও ঐতিহ্যমণ্ডিত। নদীমাতৃক এই ভূখণ্ডে বহু প্রাচীন সভ্যতার নিদর্শন বিদ্যমান।"
    rows = [
        {"id": "doc1", "text": clean_text},
        {"id": "doc1", "text": clean_text},  # duplicate ID
        {"id": "doc2", "text": "This is entirely english text rejected by quality."},  # reject
    ]
    with open(input_train, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = clean_train_file(
        input_file=input_train,
        output_file=output_train,
        pages_path=pages_p,
        egen_path=egen_p,
        tok_sample_path=tok_sample_p,
    )

    assert report["docs_in"] == 3
    assert report["docs_out"] == 1
    assert report["total_purged_docs"] == 2
    assert report["purged_counts"]["duplicate_ids"] == 1
    assert report["purged_counts"]["quality_rejected"] == 1
    assert report["plant_audit"]["all_passed"] is True
    assert output_train.exists()

