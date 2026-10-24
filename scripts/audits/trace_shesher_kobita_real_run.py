"""
scripts/audits/trace_shesher_kobita_real_run.py : Execute real Shesher Kobita text through actual builder pipeline.
Evaluates:
  Case 1: Document containing explicit title 'শেষের কবিতা'
  Case 2: Pure uncredited interior passage from Shesher Kobita (dialogue/poem, no author/title mentioned)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.build_fw2_100m_sample import NARROW_TAGORE_KEYWORDS, normalize_for_filter
from src.rgpt.data.leakage import CrossCorpusLeakageDetector
from src.rgpt.text.clean import clean_pipeline
from scripts.audits.audit_leakage_plants import load_heldout_target


def test_builder_path_on_shesher_kobita() -> None:
    # 1. Load Shesher Kobita pages from raw Wikisource
    sk_pages = []
    with open("data/raw/wikisource/pages.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "শেষের কবিতা" in row.get("prefix", ""):
                sk_pages.append(row)

    print(f"Loaded {len(sk_pages)} raw pages of Shesher Kobita.")
    assert len(sk_pages) > 0

    # Build real leakage detector
    print("Building CrossCorpusLeakageDetector on held-out targets...")
    detector = CrossCorpusLeakageDetector(
        target_texts=load_heldout_target(),
        ngram_size=5,
        window_size=30,
        threshold_distinct_ngrams=3,
    )

    # Sample Case 1: Page with title / keyword
    page_with_title = next(p for p in sk_pages if "শেষের কবিতা" in p["raw_text"])

    # Sample Case 2: Pure interior page with NO keyword in raw text
    page_interior = next(
        p for p in sk_pages
        if not any(kw in normalize_for_filter(p["raw_text"]) for kw in NARROW_TAGORE_KEYWORDS)
        and len(p.get("clean_text", "")) > 300
    )

    cases = [
        ("Case 1 (Explicit Metadata Page)", page_with_title["raw_text"]),
        ("Case 2a (Raw Wikitext Interior Page)", page_interior["raw_text"]),
        ("Case 2b (Cleaned Uncredited Interior Passage)", page_interior["clean_text"]),
    ]

    print("\n" + "=" * 60)
    print("Tracing Shesher Kobita Through Actual Builder Code Path")
    print("=" * 60)

    for case_name, text in cases:
        print(f"\n--- Testing {case_name} ---")
        print(f"Raw Text Sample: {repr(text[:90])}...")

        # Step 1: E-gen check (doc_id simulated as non-egen)
        doc_id = "test_doc_001"
        egen_ids = {"egen_mock_id"}
        if doc_id in egen_ids:
            print("  Check 1 (E-gen Disjointness): FIRED -> Excluded by Doc ID.")
            continue
        else:
            print("  Check 1 (E-gen Disjointness): PASSED (Doc ID not in E-gen set).")

        # Step 2: Keyword filter
        norm_text = normalize_for_filter(text)
        matched_kw = [kw for kw in NARROW_TAGORE_KEYWORDS if kw in norm_text]
        if matched_kw:
            print(f"  Check 2 (Keyword Filter): FIRED -> Excluded by keyword(s): {matched_kw}")
            continue
        else:
            print("  Check 2 (Keyword Filter): PASSED (No narrow keywords present).")

        # Step 3: Clean pipeline
        clean_text, passes, metrics = clean_pipeline(text, threshold=0.8)
        if not passes or not clean_text or len(clean_text) < 100:
            print(f"  Check 3 (Clean Pipeline Quality): FIRED -> Discarded (passes={passes}, len={len(clean_text)}).")
            continue
        else:
            print(f"  Check 3 (Clean Pipeline Quality): PASSED (Cleaned len={len(clean_text):,}, Bengali ratio={metrics.get('bengali_ratio', 0.9):.2%}).")

        # Step 4: Cross-corpus leakage detector
        has_leakage = detector.contains_leakage(clean_text)
        if has_leakage:
            print("  Check 4 (Cross-Corpus Leakage Detector): FIRED -> Purged! (Found >= 3 matching 5-grams in 30-word span).")
        else:
            print("  Check 4 (Cross-Corpus Leakage Detector): PASSED -> Leaked into dataset!")


if __name__ == "__main__":
    test_builder_path_on_shesher_kobita()
