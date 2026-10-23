"""
scripts/build_fw2_100m_sample.py : Stream >=100M characters of FineWeb-2 Stage 1 (ben_Beng)
with Tagore excluded using narrow keyword filter, ensuring strict disjointness from E-gen.
"""

import hashlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    out_dir = Path("data/cleaned/fineweb2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "fw2_tokenizer_sample_100m.txt"

    # 1. Load E-gen doc IDs to ensure strict disjointness
    egen_ids = set()
    egen_file = Path("data/raw/fineweb2/egen.jsonl")
    if egen_file.exists():
        with open(egen_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "id" in row:
                    egen_ids.add(row["id"])
    print(f"[E-gen] Loaded {len(egen_ids):,} held-out E-gen doc IDs to exclude.")

    # 2. Connect to FineWeb-2 train/0.parquet
    url = "https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2/parquet/ben_Beng/train/0.parquet"
    print(f"[FineWeb-2] Connecting to train/0.parquet via fsspec...")
    contact = os.environ.get("RGPT_CONTACT", "asmitsarkar@users.noreply.github.com")
    fs, path = fsspec.core.url_to_fs(url, headers={"User-Agent": f"RabindricGPT/1.0 ({contact})"})

    sampled_doc_ids = []
    total_chars = 0
    total_bytes = 0
    doc_count = 0
    tagore_excluded = 0
    egen_excluded = 0

    t0 = time.time()
    with fs.open(path, "rb") as f_in, open(out_file, "w", encoding="utf-8", newline="\n") as f_out:
        pf = pq.ParquetFile(f_in)
        num_rg = pf.num_row_groups
        print(f"[FineWeb-2] Parquet opened successfully. Total row groups: {num_rg}")

        for rg_idx in range(num_rg):
            tbl = pf.read_row_group(rg_idx, columns=["id", "text"])
            ids = tbl["id"].to_pylist()
            texts = tbl["text"].to_pylist()

            for doc_id, raw_text in zip(ids, texts):
                if doc_id in egen_ids:
                    egen_excluded += 1
                    continue

                norm_text = normalize_for_filter(raw_text)
                if any(kw in norm_text for kw in NARROW_TAGORE_KEYWORDS):
                    tagore_excluded += 1
                    continue

                clean_text, passes, metrics = clean_pipeline(raw_text, threshold=0.8)
                if not passes or not clean_text or len(clean_text) < 100:
                    continue

                f_out.write(clean_text + "\n\n")
                doc_bytes = len(clean_text.encode("utf-8"))
                doc_chars = len(clean_text)
                total_chars += doc_chars
                total_bytes += doc_bytes
                doc_count += 1
                sampled_doc_ids.append(doc_id)

                if total_chars >= target_chars:
                    break

            elapsed = time.time() - t0
            print(
                f"  RG {rg_idx:02d}/{num_rg}: {doc_count:,} docs | {total_chars:,} chars "
                f"({total_chars / 1e6:.1f}M / 100M) | {total_chars / elapsed:.0f} chars/s"
            )
            if total_chars >= target_chars:
                print(f"[FineWeb-2] Target volume reached: {total_chars:,} chars ({doc_count:,} docs).")
                break

    print(f"Finished sampling. Excluded: {egen_excluded:,} E-gen, {tagore_excluded:,} Tagore.")
    print(f"Saved {doc_count:,} documents ({total_chars:,} chars, {total_bytes:,} bytes) to {out_file}")


if __name__ == "__main__":
    main()
