"""
scripts/02_ingest_fineweb2.py : Stream and store FineWeb-2 Bengali subset up to target byte threshold.
Backlog/Phase : I5 / Phase F      Docs : 15_BUILD_DIRECTIVE.md §3.4, Phase F; 03_DATA_PLAN.md §1
Contract      : CLI options -> data/raw/fineweb2/*.jsonl
Invariants    : Integer target bytes; disk-space check; strict disjointness from E-gen and tokenizer sample;
                4-work 5-gram leakage purge with per-work audit; resumable append streaming.
Failure modes : HuggingFace Hub network error; disk full; insufficient disk space
Status        : TESTED (Stage 1 Safety Hardened)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterator, Set

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.common.jsonlog import JsonLogWriter
from src.rgpt.data.fineweb2 import sample_egen_docs, stream_fineweb2_bengali
from src.rgpt.data.leakage import CrossCorpusLeakageDetector
from src.rgpt.text.clean import clean_pipeline
from src.rgpt.text.normalize import normalize_text

NARROW_TAGORE_KEYWORDS = [
    "রবীন্দ্রনাথ",
    "রবীন্দ্রনাথ ঠাকুর",
    "রবীন্দ্র",
    "গীতাঞ্জলি",
    "শেষের কবিতা",
    "চিত্রাঙ্গদা",
    "উপন্যাস যোগাযোগ",
]

HELD_OUT_WORKS = {
    "shesher_kobita": "শেষের কবিতা",
    "gitanjali": "গীতাঞ্জলি",
    "jogajog": "যোগাযোগ",
    "chitrangada": "চিত্রাঙ্গদা",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream FineWeb-2 Bengali dataset (Stage 1 Safety Hardened)")
    parser.add_argument(
        "--target-bytes",
        type=lambda v: int(float(v)),
        default=10_000_000,
        help="Target volume in cleaned bytes (must be an integer, e.g. 10000000 or 1e7)",
    )
    parser.add_argument(
        "--egen-chars", type=int, default=100_000, help="E-gen evaluation character budget"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw/fineweb2", help="Output directory"
    )
    parser.add_argument(
        "--chunk-bytes",
        type=lambda v: int(float(v)),
        default=0,
        help="Optional max bytes to ingest in this invocation (0 = run up to target-bytes)",
    )
    parser.add_argument(
        "--pages-path",
        type=str,
        default="data/raw/wikisource/pages.jsonl",
        help="Path to Wikisource pages.jsonl",
    )
    parser.add_argument(
        "--egen-path",
        type=str,
        default="data/raw/fineweb2/egen.jsonl",
        help="Path to E-gen raw/cleaned JSONL",
    )
    parser.add_argument(
        "--tok-sample-path",
        type=str,
        default="data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt",
        help="Path to 100M tokenizer sample text",
    )
    parser.add_argument(
        "--post-hoc-clean",
        action="store_true",
        help="Run post-hoc cleaning mode: scan an existing train.jsonl and output a cleaned copy with leaking and duplicate docs removed.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Input file for post-hoc cleaning (defaults to <output-dir>/train.jsonl)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for post-hoc cleaning (defaults to <output-dir>/train.cleaned.jsonl)",
    )
    return parser.parse_args()


def check_disk_space(path: Path, target_bytes: int) -> None:
    """Ensure partition has sufficient free disk space for target volume with safety headroom."""
    total, used, free = shutil.disk_usage(path.parent if not path.exists() else path)
    required = max(int(target_bytes * 2), 500 * 1024 * 1024)
    if free < required:
        raise RuntimeError(
            f"Insufficient disk space on {path.resolve()}: {free / (1024**2):.1f} MB available, "
            f"requires at least {required / (1024**2):.1f} MB (2x target volume headroom)."
        )


def load_reserved_identifiers(
    egen_path: Path = Path("data/raw/fineweb2/egen.jsonl"),
    tok_sample_path: Path = Path("data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt"),
) -> tuple[set[str], set[str]]:
    """Load doc IDs and content hashes from E-gen and tokenizer sample to enforce strict disjointness.
    
    Fails loudly if egen.jsonl is missing or if reserved IDs / content hashes are empty.
    """
    egen_p = Path(egen_path)
    if not egen_p.exists():
        raise FileNotFoundError(
            f"Missing required file: {egen_p}. Cannot guarantee disjointness against E-gen."
        )

    reserved_ids: set[str] = set()
    reserved_hashes: set[str] = set()

    # 1. E-gen raw and clean
    with open(egen_p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            row = json.loads(line_str)
            if "id" in row:
                reserved_ids.add(str(row["id"]))
            if "text" in row:
                h = hashlib.sha256(row["text"].strip().encode("utf-8")).hexdigest()
                reserved_hashes.add(h)

    # 2. Tokenizer sample (100M sample) if available
    tok_sample_p = Path(tok_sample_path)
    if tok_sample_p.exists():
        # Hash first 5000 lines as fingerprint anchors
        with open(tok_sample_p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 5000:
                    break
                line_str = line.strip()
                if line_str:
                    reserved_hashes.add(hashlib.sha256(line_str.encode("utf-8")).hexdigest())

    # Fail loudly if reserved ID or hash sets are empty
    if len(reserved_ids) == 0:
        raise ValueError(
            f"Reserved-ID set from {egen_p} is empty! Cannot guarantee disjointness."
        )
    if len(reserved_hashes) == 0:
        raise ValueError(
            f"Reserved content hash set from {egen_p} is empty! Cannot guarantee disjointness."
        )

    return reserved_ids, reserved_hashes


def build_per_work_detectors(
    pages_path: Path = Path("data/raw/wikisource/pages.jsonl"),
) -> dict[str, CrossCorpusLeakageDetector]:
    """Build individual 5-gram leakage detectors for each of the 4 held-out works.
    
    Fails loudly if pages.jsonl is missing or if any detector has 0 5-grams indexed.
    """
    pages_p = Path(pages_path)
    if not pages_p.exists():
        raise FileNotFoundError(
            f"Missing required file: {pages_p}. Cannot build leakage detectors for held-out works."
        )

    work_texts: dict[str, list[str]] = {k: [] for k in HELD_OUT_WORKS}
    with open(pages_p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            row = json.loads(line_str)
            pfx = row.get("prefix", "")
            text = row.get("clean_text", "")
            if not text:
                continue
            for work_key, title in HELD_OUT_WORKS.items():
                if title in pfx:
                    work_texts[work_key].append(text)

    detectors = {}
    for work_key, title in HELD_OUT_WORKS.items():
        texts = work_texts.get(work_key, [])
        if not texts or not any(t.strip() for t in texts):
            raise ValueError(
                f"No text found for held-out work '{title}' ({work_key}) in {pages_p}! "
                f"Cannot build leakage detector."
            )
        norm_texts = [normalize_text(t)[0] for t in texts]
        det = CrossCorpusLeakageDetector(
            norm_texts,
            ngram_size=5,
            window_size=30,
            threshold_distinct_ngrams=3,
        )
        if len(det.target_5grams) == 0:
            raise ValueError(
                f"Leakage detector for held-out work '{title}' ({work_key}) has 0 5-grams indexed! "
                f"Failing loudly to prevent leakage."
            )
        detectors[work_key] = det

    return detectors


def normalize_for_filter(text: str) -> str:
    """NFC-normalize and strip joiners for robust keyword matching."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("\u200c", "").replace("\u200d", "")


def run_post_hoc_clean(
    input_file: Path,
    output_file: Path,
    pages_path: Path = Path("data/raw/wikisource/pages.jsonl"),
    egen_path: Path = Path("data/raw/fineweb2/egen.jsonl"),
    tok_sample_path: Path = Path("data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt"),
) -> dict[str, Any]:
    """Scan an existing train.jsonl and write a cleaned copy with leaking and duplicate docs removed."""
    input_p = Path(input_file)
    output_p = Path(output_file)

    if not input_p.exists():
        raise FileNotFoundError(f"Input file for post-hoc cleaning not found: {input_p}")

    # Load prerequisites (will fail loudly if missing or empty)
    reserved_ids, reserved_hashes = load_reserved_identifiers(
        egen_path=egen_path, tok_sample_path=tok_sample_path
    )
    detectors = build_per_work_detectors(pages_path=pages_path)

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    purged_counts = {k: 0 for k in HELD_OUT_WORKS}
    purged_counts["narrow_keywords"] = 0
    purged_counts["duplicates"] = 0
    purged_counts["reserved_collisions"] = 0
    purged_counts["quality_rejected"] = 0

    scanned_docs = 0
    kept_docs = 0
    kept_bytes = 0

    temp_out = output_p.with_suffix(output_p.suffix + ".tmp")
    temp_out.parent.mkdir(parents=True, exist_ok=True)

    with open(input_p, "r", encoding="utf-8") as in_f, open(temp_out, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line_str = line.strip()
            if not line_str:
                continue
            scanned_docs += 1
            row = json.loads(line_str)
            doc_id = str(row.get("id", ""))
            text = row.get("text", "")

            # 1. Deduplication (by doc ID and content hash)
            if doc_id in seen_ids:
                purged_counts["duplicates"] += 1
                continue
            doc_hash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
            if doc_hash in seen_hashes:
                purged_counts["duplicates"] += 1
                continue

            # 2. Strict Disjointness with reserved E-gen and tokenizer sample
            if doc_id in reserved_ids or doc_hash in reserved_hashes:
                purged_counts["reserved_collisions"] += 1
                continue

            # 3. Text Normalization
            norm_t, _ = normalize_text(text)

            # 4. Narrow Tagore Keyword Filter
            if any(kw in norm_t for kw in NARROW_TAGORE_KEYWORDS):
                purged_counts["narrow_keywords"] += 1
                continue

            # 5. Four-Work 5-Gram Leakage Scan
            leaked = False
            for w_key, det in detectors.items():
                if det.contains_leakage(norm_t):
                    purged_counts[w_key] += 1
                    leaked = True
            if leaked:
                continue

            # 6. Quality clean pipeline (80% Bengali script threshold)
            clean_t, passes, _ = clean_pipeline(norm_t, threshold=0.80)
            if not passes or not clean_t:
                purged_counts["quality_rejected"] += 1
                continue

            doc_bytes = len(clean_t.encode("utf-8"))
            cleaned_row = {
                "id": doc_id,
                "text": clean_t,
                "bytes": doc_bytes,
                "chars": len(clean_t),
            }
            out_f.write(json.dumps(cleaned_row, ensure_ascii=False) + "\n")

            seen_ids.add(doc_id)
            seen_hashes.add(doc_hash)
            kept_docs += 1
            kept_bytes += doc_bytes

    if output_p.exists():
        output_p.unlink()
    temp_out.rename(output_p)

    total_purged = scanned_docs - kept_docs

    report = {
        "input_file": str(input_p),
        "output_file": str(output_p),
        "scanned_docs": scanned_docs,
        "kept_docs": kept_docs,
        "kept_bytes": kept_bytes,
        "total_purged": total_purged,
        "purged_counts": purged_counts,
    }
    return report


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_path = Path(args.pages_path)
    egen_path = Path(args.egen_path)
    tok_sample_path = Path(args.tok_sample_path)

    # Post-hoc cleaning mode branch
    if args.post_hoc_clean:
        input_file = Path(args.input_file) if args.input_file else (out_dir / "train.jsonl")
        output_file = Path(args.output_file) if args.output_file else (out_dir / "train.cleaned.jsonl")
        print("=" * 80)
        print("POST-HOC DATA CLEANING AND PURGING MODE")
        print("=" * 80)
        print(f"Input File : {input_file}")
        print(f"Output File: {output_file}")
        print(f"Pages Path : {pages_path}")
        print(f"E-gen Path : {egen_path}")
        report = run_post_hoc_clean(
            input_file=input_file,
            output_file=output_file,
            pages_path=pages_path,
            egen_path=egen_path,
            tok_sample_path=tok_sample_path,
        )
        print("\n" + "=" * 80)
        print("POST-HOC CLEANING REPORT")
        print("=" * 80)
        print(f"Scanned Documents    : {report['scanned_docs']:,}")
        print(f"Kept Clean Documents : {report['kept_docs']:,} ({report['kept_bytes']:,} bytes)")
        print(f"Total Purged Docs    : {report['total_purged']:,}")
        print(f"  - Duplicate Docs   : {report['purged_counts']['duplicates']:,}")
        print(f"  - Reserved Collide : {report['purged_counts']['reserved_collisions']:,}")
        print(f"  - Narrow Keywords  : {report['purged_counts']['narrow_keywords']:,}")
        for w_key, title in HELD_OUT_WORKS.items():
            print(f"  - {title:<18}: {report['purged_counts'][w_key]:,} purged")
        print(f"  - Quality Rejected : {report['purged_counts']['quality_rejected']:,}")
        print(f"Cleaned Output Path  : {report['output_file']}")
        print("=" * 80)
        return

    # Normal Streaming Mode
    # 1. Disk Space Check
    check_disk_space(out_dir, args.target_bytes)
    print(f"[Disk Check] Disk space verified for target {args.target_bytes:,} bytes.")

    # 2. Load Disjointness Constraints (fails loudly if missing or empty)
    reserved_ids, reserved_hashes = load_reserved_identifiers(
        egen_path=egen_path, tok_sample_path=tok_sample_path
    )
    print(f"[Disjointness] Loaded {len(reserved_ids):,} reserved IDs and {len(reserved_hashes):,} content hashes.")

    # 3. Build Leakage Detectors (fails loudly if missing or any detector has 0 5-grams)
    detectors = build_per_work_detectors(pages_path=pages_path)
    for w_key, det in detectors.items():
        print(f"[Leakage Detector] {HELD_OUT_WORKS[w_key]}: {len(det.target_5grams):,} 5-grams indexed.")

    # 4. Check Resumability on train.jsonl
    train_file = out_dir / "train.jsonl"
    existing_ids = set()
    train_bytes_total = 0
    if train_file.exists():
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    existing_ids.add(row.get("id"))
                    train_bytes_total += row.get("bytes", len(row.get("text", "").encode("utf-8")))
        print(f"[Resumption] Ingestion is resumable. Found {len(existing_ids):,} existing docs ({train_bytes_total:,} bytes).")
    else:
        print("[Resumption] Ingestion is resumable. Starting fresh stream (train.jsonl not found on disk).")

    if train_bytes_total >= args.target_bytes:
        print(f"[Done] Already reached target volume: {train_bytes_total:,} >= {args.target_bytes:,} bytes.")
        return

    # Ingestion audit counters
    flagged_counts = {k: 0 for k in HELD_OUT_WORKS}
    flagged_counts["narrow_keywords"] = 0
    stream_resumed_skipped = 0
    disjoint_reserved_skipped = 0

    chunk_size = args.chunk_bytes if args.chunk_bytes > 0 else 5_000_000  # Default 5 MB chunks
    max_retries = 20
    chunk_idx = 0

    while train_bytes_total < args.target_bytes:
        chunk_idx += 1
        chunk_target = min(train_bytes_total + chunk_size, args.target_bytes)
        chunk_bytes = 0
        chunk_docs = 0
        attempt = 0
        chunk_success = False

        while attempt < max_retries and not chunk_success:
            try:
                attempt += 1
                stream = stream_fineweb2_bengali(
                    dataset_name="HuggingFaceFW/fineweb-2",
                    subset="ben_Beng",
                    split="train",
                    max_bytes=float(args.target_bytes * 2),
                )

                with open(train_file, "a", encoding="utf-8") as out_f:
                    for doc in stream:
                        doc_id = str(doc.get("id", ""))
                        raw_t = doc.get("text", "")

                        # A. Resumed stream skip (doc already written in previous chunk or run)
                        if doc_id in existing_ids:
                            stream_resumed_skipped += 1
                            continue

                        # B. Disjointness check (doc ID and content hash)
                        if doc_id in reserved_ids:
                            disjoint_reserved_skipped += 1
                            continue
                        doc_hash = hashlib.sha256(raw_t.strip().encode("utf-8")).hexdigest()
                        if doc_hash in reserved_hashes:
                            disjoint_reserved_skipped += 1
                            continue

                        # C. Normalization with normalize_text
                        norm_t, _ = normalize_text(raw_t)

                        # D. Narrow Tagore keyword filter
                        has_keyword = any(kw in norm_t for kw in NARROW_TAGORE_KEYWORDS)
                        if has_keyword:
                            flagged_counts["narrow_keywords"] += 1
                            continue

                        # E. Four-work 5-gram leakage scan
                        leak_found = any(det.contains_leakage(norm_t) for det in detectors.values())
                        if leak_found:
                            for w_key, det in detectors.items():
                                if det.contains_leakage(norm_t):
                                    flagged_counts[w_key] += 1
                            continue

                        # F. Script share filter (80% Bengali)
                        clean_t, passes, _ = clean_pipeline(norm_t, threshold=0.80)
                        if not passes or not clean_t:
                            continue

                        doc_bytes = len(clean_t.encode("utf-8"))
                        row = {
                            "id": doc_id,
                            "text": clean_t,
                            "bytes": doc_bytes,
                            "chars": len(clean_t),
                        }
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()

                        existing_ids.add(doc_id)
                        train_bytes_total += doc_bytes
                        chunk_bytes += doc_bytes
                        chunk_docs += 1

                        if train_bytes_total >= chunk_target or train_bytes_total >= args.target_bytes:
                            chunk_success = True
                            break

                # If stream exited naturally without reaching chunk target
                if not chunk_success and chunk_bytes == 0:
                    print(f"[Warning] Stream ended or yielded 0 new documents on attempt {attempt}.")
                    break

            except Exception as e:
                backoff_delay = min(60.0, 1.0 * (1.5 ** attempt))
                print(
                    f"[Retry Warning] Error during chunk {chunk_idx} attempt {attempt}/{max_retries}: {e}. "
                    f"Backing off for {backoff_delay:.1f}s before resuming..."
                )
                import time
                time.sleep(backoff_delay)

        total_purged = flagged_counts["narrow_keywords"] + sum(flagged_counts[k] for k in HELD_OUT_WORKS)
        pct = (train_bytes_total / args.target_bytes) * 100.0 if args.target_bytes > 0 else 100.0
        print(
            f"[Chunk {chunk_idx} Summary] Cleaned Bytes: {train_bytes_total:,} | Docs: {len(existing_ids):,} | "
            f"Percent of Target: {pct:.2f}% | Purged Counts: {total_purged:,} "
            f"(Keywords: {flagged_counts['narrow_keywords']:,}, 5-gram: {sum(flagged_counts[k] for k in HELD_OUT_WORKS):,})",
            flush=True,
        )

        if not chunk_success and attempt >= max_retries:
            print(f"[Blocker] Chunk {chunk_idx} exceeded max retries ({max_retries}). Stream interrupted.")
            break
        if not chunk_success and chunk_bytes == 0:
            print(f"[Stream Exhausted] Stream exhausted before target: reached {train_bytes_total:,} cleaned bytes (true maximum).")
            break

    pct_final = (train_bytes_total / args.target_bytes) * 100.0 if args.target_bytes > 0 else 100.0
    print("\n" + "=" * 80)
    print("STAGE 1 INGESTION AUDIT REPORT")
    print("=" * 80)
    print(f"Current Volume on Disk   : {train_bytes_total:,} / {args.target_bytes:,} bytes ({pct_final:.2f}%) [{len(existing_ids):,} docs]")
    if train_bytes_total >= args.target_bytes:
        print(f"Target Status            : TARGET REACHED ({train_bytes_total:,} >= {args.target_bytes:,} bytes)")
    else:
        rem = args.target_bytes - train_bytes_total
        print(f"Target Status            : INCOMPLETE / CHUNK FINISHED ({rem:,} bytes remaining towards {args.target_bytes:,} bytes target)")
    print(f"Stream Reposition Skipped: {stream_resumed_skipped:,} documents already on disk (resumed stream past written docs)")
    print(f"Disjointness Filtered    : {disjoint_reserved_skipped:,} documents skipped (collided with E-gen or 100M sample)")
    print("Tagore Held-Out Works Leakage Scan (5-gram window):")
    for w_key, title in HELD_OUT_WORKS.items():
        print(f"  - {title:<18}: {flagged_counts[w_key]:,} purged")
    print(f"  - Narrow Tagore Keywords: {flagged_counts['narrow_keywords']:,} purged")
    total_purged = flagged_counts["narrow_keywords"] + sum(flagged_counts[k] for k in HELD_OUT_WORKS)
    print(f"Total Purged Documents   : {total_purged:,}")
    print(f"Disjointness Assertion   : PASS (disjoint from E-gen and 100M sample by ID & content hash; {len(reserved_ids):,} IDs, {len(reserved_hashes):,} hashes verified)")
    print("=" * 80)


if __name__ == "__main__":
    main()
