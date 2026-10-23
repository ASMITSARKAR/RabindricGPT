"""
scripts/rebuild_clean_lf_tokenizer_sample.py : Build pure Unix-LF Stage-1 FineWeb-2 sample
using narrow Tagore filter, purging Shesher Kobita leak, and ensuring b"\\r\\n" == 0.
"""

import hashlib
import json
import os
import random
import sys
import time
import unicodedata
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.data.leakage import CrossCorpusLeakageDetector, tokenize_bengali_words
from src.rgpt.text.clean import clean_pipeline

NARROW_TAGORE_KEYWORDS = [
    "রবীন্দ্রনাথ ঠাকুর",
    "রবীন্দ্রনাথ ঠাকুরের",
    "রবীন্দ্রনাথের",
    "রবীন্দ্রসঙ্গীত",
    "রবীন্দ্র-সংগীত",
    "ভানুসিংহ",
    "শেষের কবিতা",
    "চোখের বালি",
    "ঘরে বাইরে",
    "রক্তকরবী",
    "গীতাঞ্জলি",
    "সোনার তরী",
    "গল্পগুচ্ছ",
    "চতুরঙ্গ",
    "নাটক ডাকঘর",
    "নৌকাডুবি",
    "গোরা উপন্যাস",
    "যোগাযোগ উপন্যাস",
    "উপন্যাস যোগাযোগ",
]


def normalize_for_filter(text: str) -> str:
    """NFC-normalize and strip ZWNJ/ZWJ before keyword matching."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("\u200c", "").replace("\u200d", "")


def main() -> None:
    import fsspec
    import pyarrow.parquet as pq

    target_chars = 100_000_000
    out_file = Path("data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt")

    # 1. Load E-gen doc IDs
    egen_ids = set()
    egen_file = Path("data/raw/fineweb2/egen.jsonl")
    if egen_file.exists():
        with open(egen_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "id" in row:
                    egen_ids.add(row["id"])
    print(f"[E-gen] Loaded {len(egen_ids):,} held-out E-gen doc IDs.")

    # 2. Build 4-work leakage detector
    val_test_texts = []
    with open("data/raw/wikisource/pages.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pfx = row.get("prefix", "")
            if any(k in pfx for k in ["শেষের কবিতা", "গীতাঞ্জলি", "যোগাযোগ", "চিত্রাঙ্গদা"]):
                val_test_texts.append(row.get("clean_text", ""))
    leak_detector = CrossCorpusLeakageDetector(
        val_test_texts, ngram_size=5, window_size=30, threshold_distinct_ngrams=3
    )
    print(f"[Leakage] Built detector with {len(leak_detector.target_5grams):,} target 5-grams.")

    # 3. Stream from FineWeb-2
    url = "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2/parquet/ben_Beng/train/0.parquet"
    print(f"[FineWeb-2] Connecting via fsspec...")
    contact = os.environ.get("RGPT_CONTACT", "asmitsarkar@users.noreply.github.com")
    fs, path = fsspec.core.url_to_fs(url, headers={"User-Agent": f"RabindricGPT/1.0 ({contact})"})

    random.seed(42)
    sampled_rgs = sorted(random.sample(range(2912), 60))
    print(f"Sampled 60 row groups uniformly across 2,912 total RGs (seed=42). First 5: {sampled_rgs[:5]}")

    total_chars = 0
    doc_count = 0
    tagore_excluded = 0
    leak_purged = 0
    sampled_ids = []

    t0 = time.time()
    with fs.open(path, "rb") as f_in, open(out_file, "w", encoding="utf-8", newline="\n") as f_out:
        pf = pq.ParquetFile(f_in)
        for rg_idx in sampled_rgs:
            tbl = pf.read_row_group(rg_idx, columns=["id", "text"])
            ids = tbl["id"].to_pylist()
            texts = tbl["text"].to_pylist()

            for doc_id, raw_text in zip(ids, texts):
                if doc_id in egen_ids:
                    continue

                norm_text = normalize_for_filter(raw_text)
                if any(kw in norm_text for kw in NARROW_TAGORE_KEYWORDS):
                    tagore_excluded += 1
                    continue

                clean_text, passes, _ = clean_pipeline(raw_text, threshold=0.8)
                if not passes or not clean_text or len(clean_text) < 100:
                    continue

                if leak_detector.contains_leakage(clean_text):
                    leak_purged += 1
                    print(f"  [LEAK PURGED] Caught leaking quote in doc {doc_id} ({len(clean_text)} chars)!")
                    continue

                f_out.write(clean_text + "\n\n")
                total_chars += len(clean_text)
                doc_count += 1
                sampled_ids.append(doc_id)

                if total_chars >= target_chars:
                    break
            if total_chars >= target_chars:
                break

    print(f"Sampling complete: {doc_count:,} docs, {total_chars:,} chars. Excluded {tagore_excluded:,} Tagore keywords, purged {leak_purged:,} leaks.")


if __name__ == "__main__":
    main()
