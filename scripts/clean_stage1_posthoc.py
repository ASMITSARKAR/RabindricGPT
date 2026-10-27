"""
scripts/clean_stage1_posthoc.py : Post-hoc streaming audit, deduplication, and
leakage purge of Stage 1 FineWeb-2 train.jsonl with full accounting and plant recall.

Enforces:
1. Low-memory line-by-line streaming.
2. Deduplication of doc IDs and SHA-256 content hashes.
3. Disjointness from E-gen and 100M tokenizer sample.
4. Narrow Tagore keyword purge with frequency accounting and example snippet collection.
5. 4-work 5-gram leakage scan (Shesher Kobita, Gitanjali, Jogajog, Chitrangada).
6. Script purity (>= 80% Bengali).
7. Normalizer invariants: zero U+FEFF, zero ZWNJ, ZWJ only in [C] + virama + ZWJ + ya.
8. 20-plant verbatim 13-gram recall test across all 4 held-out works.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.data.leakage import CrossCorpusLeakageDetector, tokenize_bengali_words
from src.rgpt.text.clean import clean_pipeline
from src.rgpt.text.normalize import normalize_text

HELD_OUT_WORKS: Dict[str, str] = {
    "shesher_kobita": "শেষের কবিতা",
    "gitanjali": "গীতাঞ্জলি",
    "jogajog": "যোগাযোগ",
    "chitrangada": "চিত্রাঙ্গদা",
}

NARROW_TAGORE_KEYWORDS = [
    "রবীন্দ্রনাথ",
    "রবীন্দ্রনাথ ঠাকুর",
    "রবীন্দ্র",
    "গীতাঞ্জলি",
    "শেষের কবিতা",
    "চিত্রাঙ্গদা",
    "উপন্যাস যোগাযোগ",
]

SYNTHETIC_BACKGROUND_DOCS = [
    "বাংলা সাহিত্যের ইতিহাস অত্যন্ত প্রাচীন ও সমৃদ্ধ। প্রাচীন যুগ হইতে আধুনিক কাল পর্যন্ত বহু সাধক কবি এখানে সাধনা করিয়াছেন।",
    "শরতের নীল আকাশে সাদা মেঘের ভেলা ভাসিয়া বেড়ায়। কাশবনে নদীর তীরে মৃদু বাতাসে কাশফুল দোলে।",
    "বাংলাদেশের অর্থনীতি মূলত কৃষিনির্ভর হইলেও সাম্প্রতিককালে তথ্যপ্রযুক্তি ও তৈরি পোশাক শিল্পে অভূতপূর্ব উন্নয়ন সাধিত হইয়াছে।",
    "শিক্ষা জাতির মেরুদণ্ড। সঠিক ও মানসম্মত শিক্ষা ছাড়া কোনো জাতি বিশ্বদরবারে মাথা উঁচু করিয়া দাঁড়াইতে পারে না।",
    "প্রাকৃতিক দুর্যোগ মোকাবেলায় উপকূলীয় বাঁধ ও সবুজ বেষ্টনী নির্মাণ অত্যন্ত জরুরি হইয়া পড়িয়াছে বলিয়া বিশেষজ্ঞরা জানাইয়াছেন।",
    "বিজ্ঞান ও প্রযুক্তির দ্রুত অগ্রগতি মানব সভ্যতার গতিপথ সম্পূর্ণ পরিবর্তন করিয়া দিয়াছে বলিয়া গবেষকগণ মনে করেন।",
    "নদীর জল শান্ত ও স্বচ্ছ। দূর দিগন্তে মাঝি তাহার নৌকা ভাসাইয়া মনের আনন্দে ভাটিয়ালি গান গাহিতেছে।",
    "স্বাস্থ্য সচেতনতা বাড়াইতে নিয়মিত শরীরচর্চা ও সুষম খাদ্য গ্রহণের প্রয়োজনীয়তা সম্পর্কে সামাজিক প্রচার চালানো হইতেছে।",
]


def load_heldout_work_texts(pages_path: Path) -> Dict[str, List[str]]:
    """Load text for the 4 held-out works from wikisource pages.jsonl."""
    pages_p = Path(pages_path)
    if not pages_p.exists():
        raise FileNotFoundError(
            f"Missing required file: {pages_p}. Cannot load held-out works."
        )

    work_pages: Dict[str, List[str]] = defaultdict(list)
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
                    work_pages[work_key].append(text)
                    break

    for work_key, title in HELD_OUT_WORKS.items():
        if not work_pages[work_key]:
            raise ValueError(
                f"No text found for held-out work '{title}' ({work_key}) in {pages_p}!"
            )

    return work_pages


def build_detectors(pages_path: Path) -> Dict[str, CrossCorpusLeakageDetector]:
    """Build individual 5-gram leakage detectors for each held-out work."""
    work_texts = load_heldout_work_texts(pages_path)
    detectors: Dict[str, CrossCorpusLeakageDetector] = {}

    for work_key, texts in work_texts.items():
        norm_texts = [normalize_text(t)[0] for t in texts]
        det = CrossCorpusLeakageDetector(
            target_texts=norm_texts,
            ngram_size=5,
            window_size=30,
            threshold_distinct_ngrams=3,
        )
        if len(det.target_5grams) == 0:
            raise ValueError(
                f"Leakage detector for held-out work '{HELD_OUT_WORKS[work_key]}' ({work_key}) has 0 5-grams indexed!"
            )
        detectors[work_key] = det

    return detectors


def load_reserved(egen_path: Path, tok_sample_path: Path) -> Tuple[Set[str], Set[str]]:
    """Load reserved doc IDs and SHA-256 hashes from E-gen and 100M sample."""
    egen_p = Path(egen_path)
    tok_sample_p = Path(tok_sample_path)

    if not egen_p.exists():
        raise FileNotFoundError(f"Missing required file: {egen_p}")
    if not tok_sample_p.exists():
        raise FileNotFoundError(f"Missing required file: {tok_sample_p}")

    reserved_ids: Set[str] = set()
    reserved_hashes: Set[str] = set()

    with open(egen_p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            row = json.loads(line_str)
            if "id" in row:
                reserved_ids.add(str(row["id"]))
            if "text" in row:
                reserved_hashes.add(hashlib.sha256(row["text"].strip().encode("utf-8")).hexdigest())

    with open(tok_sample_p, "r", encoding="utf-8") as f:
        doc_buf: List[str] = []
        doc_count = 0
        for line in f:
            line_str = line.strip()
            if line_str:
                doc_buf.append(line_str)
            else:
                if doc_buf:
                    doc_text = "\n".join(doc_buf)
                    reserved_hashes.add(hashlib.sha256(doc_text.encode("utf-8")).hexdigest())
                    for p in doc_buf:
                        if len(p) >= 40:
                            reserved_hashes.add(hashlib.sha256(p.encode("utf-8")).hexdigest())
                    doc_buf = []
                    doc_count += 1
                    if doc_count >= 10000:
                        break
        if doc_buf and doc_count < 10000:
            doc_text = "\n".join(doc_buf)
            reserved_hashes.add(hashlib.sha256(doc_text.encode("utf-8")).hexdigest())

    if len(reserved_ids) == 0:
        raise ValueError(f"Reserved-ID set from {egen_p} is empty!")
    if len(reserved_hashes) == 0:
        raise ValueError(f"Reserved content hash set from {egen_p} is empty!")

    return reserved_ids, reserved_hashes


def run_plant_recall_audit(
    detectors: Dict[str, CrossCorpusLeakageDetector],
    work_texts: Dict[str, List[str]],
    n_plants_per_work: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify >= 95% recall for 20 planted verbatim 13-grams per work."""
    random.seed(seed)
    recall_results = {}

    for work_key, title in HELD_OUT_WORKS.items():
        texts = work_texts.get(work_key, [])
        all_words = tokenize_bengali_words("\n\n".join(texts))
        if len(all_words) < 30:
            raise ValueError(f"Insufficient text for work '{title}' ({len(all_words)} words)")

        step = max(1, (len(all_words) - 15) // n_plants_per_work)
        planted_hits = 0

        for i in range(n_plants_per_work):
            start = i * step
            gram_words = all_words[start : start + 13]
            planted_phrase = " ".join(gram_words)

            bg = SYNTHETIC_BACKGROUND_DOCS[i % len(SYNTHETIC_BACKGROUND_DOCS)]
            bg_words = tokenize_bengali_words(bg)
            insert_idx = len(bg_words) // 2
            planted_doc = " ".join(bg_words[:insert_idx] + gram_words + bg_words[insert_idx:])

            if detectors[work_key].contains_leakage(planted_doc):
                planted_hits += 1

        rec = (planted_hits / n_plants_per_work) * 100.0
        recall_results[work_key] = {
            "title": title,
            "plants": n_plants_per_work,
            "hits": planted_hits,
            "recall_pct": rec,
            "passed": rec >= 95.0,
        }

    total_plants = n_plants_per_work * len(HELD_OUT_WORKS)
    total_hits = sum(r["hits"] for r in recall_results.values())
    total_recall = (total_hits / total_plants) * 100.0 if total_plants > 0 else 0.0

    return {
        "results": recall_results,
        "total_plants": total_plants,
        "total_hits": total_hits,
        "total_recall": total_recall,
        "all_passed": all(r["passed"] for r in recall_results.values()),
    }


def verify_normalizer_invariants(text: str) -> Tuple[bool, str]:
    """Verify zero U+FEFF, zero ZWNJ, and ZWJ only in [C]+virama+ZWJ+ya."""
    if "\ufeff" in text:
        return False, "U+FEFF (BOM) found"
    if "\u200c" in text:
        return False, "ZWNJ found"

    valid_zwj = re.compile(r"[\u0995-\u09b9\u09dc-\u09df]\u09cd\u200d\u09af")
    total_zwj = text.count("\u200d")
    valid_count = len(valid_zwj.findall(text))
    if total_zwj != valid_count:
        return False, f"Invalid ZWJ: total={total_zwj}, valid={valid_count}"
    return True, "OK"


def clean_train_file(
    input_file: Path,
    output_file: Path,
    pages_path: Path,
    egen_path: Path,
    tok_sample_path: Path,
) -> Dict[str, Any]:
    """Streaming, low-memory post-hoc cleaning of train.jsonl."""
    input_p = Path(input_file)
    output_p = Path(output_file)

    if not input_p.exists():
        raise FileNotFoundError(f"Input file not found: {input_p}")

    print("=" * 80)
    print("STAGE 1 POST-HOC DATA CLEANING AND VERIFICATION")
    print("=" * 80)
    print(f"Input File      : {input_p.resolve()}")
    print(f"Output File     : {output_p.resolve()}")
    print(f"Pages Path      : {pages_path.resolve()}")
    print(f"E-gen Path      : {egen_path.resolve()}")
    print(f"Tok Sample Path : {tok_sample_path.resolve()}")

    # 1. Build Detectors & Load Reserved IDs/Hashes
    detectors = build_detectors(pages_path)
    work_texts = load_heldout_work_texts(pages_path)
    reserved_ids, reserved_hashes = load_reserved(egen_path, tok_sample_path)

    print(f"\n[Detectors] Built 4 held-out leakage detectors:")
    for w_key, det in detectors.items():
        print(f"  - {HELD_OUT_WORKS[w_key]:<18}: {len(det.target_5grams):,} 5-grams indexed")
    print(f"[Disjointness] Loaded {len(reserved_ids):,} reserved IDs and {len(reserved_hashes):,} reserved content hashes.")

    # 2. Run 20-Plant Recall Audit
    print("\n[Audit] Running 20-plant verbatim 13-gram recall audit across 4 held-out works...")
    plant_audit = run_plant_recall_audit(detectors, work_texts, n_plants_per_work=20, seed=42)
    print(f"  Plant Recall Results (Target: >= 95.0%):")
    for w_key, r in plant_audit["results"].items():
        st = "PASS" if r["passed"] else "FAIL"
        print(f"  - {r['title']:<18}: {r['hits']}/{r['plants']} detected ({r['recall_pct']:.1f}%) [{st}]")
    print(f"  Overall Plant Recall: {plant_audit['total_hits']}/{plant_audit['total_plants']} ({plant_audit['total_recall']:.1f}%) [{'PASS' if plant_audit['all_passed'] else 'FAIL'}]")
    if not plant_audit["all_passed"]:
        raise RuntimeError("Plant recall fell below 95% threshold! Aborting clean.")

    # Fast short-circuit 5-gram union set
    all_target_5grams: Set[Tuple[str, ...]] = set()
    for det in detectors.values():
        all_target_5grams.update(det.target_5grams)

    # 3. Stream Input File
    print(f"\n[Streaming] Scanning {input_p.name}...")
    seen_ids: Set[str] = set()
    seen_hashes: Set[str] = set()

    docs_in = 0
    bytes_in = 0
    docs_out = 0
    bytes_out = 0

    purged_counts = {
        "duplicate_ids": 0,
        "duplicate_hashes": 0,
        "reserved_id_collisions": 0,
        "reserved_hash_collisions": 0,
        "narrow_keywords": 0,
        "shesher_kobita": 0,
        "gitanjali": 0,
        "jogajog": 0,
        "chitrangada": 0,
        "quality_rejected": 0,
        "invariant_violations": 0,
    }

    keyword_frequency: Dict[str, int] = defaultdict(int)
    purged_snippets: List[Dict[str, str]] = []

    temp_out = output_p.with_suffix(output_p.suffix + ".tmp")
    temp_out.parent.mkdir(parents=True, exist_ok=True)

    with open(input_p, "r", encoding="utf-8") as in_f, open(temp_out, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line_str = line.strip()
            if not line_str:
                continue

            docs_in += 1
            line_bytes = len(line_str.encode("utf-8")) + 1
            bytes_in += line_bytes

            row = json.loads(line_str)
            doc_id = str(row.get("id", ""))
            raw_text = row.get("text", "")

            # A. Deduplication by ID
            if doc_id in seen_ids:
                purged_counts["duplicate_ids"] += 1
                continue

            # B. Deduplication by Content Hash
            doc_hash = hashlib.sha256(raw_text.strip().encode("utf-8")).hexdigest()
            if doc_hash in seen_hashes:
                purged_counts["duplicate_hashes"] += 1
                continue

            # C. Disjointness check against E-gen & 100M sample
            if doc_id in reserved_ids:
                purged_counts["reserved_id_collisions"] += 1
                continue
            if doc_hash in reserved_hashes:
                purged_counts["reserved_hash_collisions"] += 1
                continue

            # D. Text Normalization
            norm_t, _ = normalize_text(raw_text)

            # E. Narrow Tagore Keyword Filter
            matched_kws = [kw for kw in NARROW_TAGORE_KEYWORDS if kw in norm_t]
            if matched_kws:
                purged_counts["narrow_keywords"] += 1
                for kw in matched_kws:
                    keyword_frequency[kw] += 1
                if len(purged_snippets) < 10:
                    preview = norm_t[:160].replace("\n", " ").strip()
                    purged_snippets.append({
                        "doc_id": doc_id,
                        "keywords": ", ".join(matched_kws),
                        "snippet": preview,
                    })
                continue

            # F. Four-Work 5-Gram Leakage Scan
            words = tokenize_bengali_words(norm_t)
            leaked = False
            if len(words) >= 5:
                # Fast check: does candidate share >= 3 distinct 5-grams with the held-out corpus?
                candidate_grams = {tuple(words[i : i + 5]) for i in range(len(words) - 4)}
                shared = candidate_grams.intersection(all_target_5grams)
                if len(shared) >= 3:
                    for w_key, det in detectors.items():
                        if det.contains_leakage(norm_t):
                            purged_counts[w_key] += 1
                            leaked = True
            if leaked:
                continue

            # G. Script Purity Filter (>= 80% Bengali)
            clean_t, passes, _ = clean_pipeline(norm_t, threshold=0.80)
            if not passes or not clean_t:
                purged_counts["quality_rejected"] += 1
                continue

            # H. Verify Normalizer Invariants on Kept Text
            inv_ok, inv_reason = verify_normalizer_invariants(clean_t)
            if not inv_ok:
                purged_counts["invariant_violations"] += 1
                continue

            clean_bytes = len(clean_t.encode("utf-8"))
            clean_row = {
                "id": doc_id,
                "text": clean_t,
                "bytes": clean_bytes,
                "chars": len(clean_t),
            }
            out_f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")

            seen_ids.add(doc_id)
            seen_hashes.add(doc_hash)
            docs_out += 1
            bytes_out += clean_bytes

            if docs_in % 25000 == 0:
                print(f"  Processed {docs_in:,} docs ({bytes_in / (1024**2):.1f} MB in) -> {docs_out:,} docs kept ({bytes_out / (1024**2):.1f} MB out)")

    # Atomic rename
    if output_p.exists():
        output_p.unlink()
    temp_out.rename(output_p)

    total_purged_docs = docs_in - docs_out

    # 4. Compute 10M Real Epochs
    # N_ne for 10M is 10,621,824 parameters; 100 * N_ne = 1,062,182,400 bytes
    n_ne_10m = 10_621_824
    target_bytes_10m = 100 * n_ne_10m
    real_epochs_10m = target_bytes_10m / bytes_out if bytes_out > 0 else 0.0

    print("\n" + "=" * 80)
    print("STAGE 1 POST-HOC CLEANING COMPLETE REPORT")
    print("=" * 80)
    print(f"Documents In            : {docs_in:,} docs")
    print(f"Bytes In                : {bytes_in:,} bytes ({bytes_in / (1024**2):.2f} MB)")
    print(f"Documents Out (Kept)    : {docs_out:,} docs")
    print(f"Bytes Out (Kept)        : {bytes_out:,} bytes ({bytes_out / (1024**2):.2f} MB)")
    print(f"Total Purged Documents  : {total_purged_docs:,} docs ({(total_purged_docs / docs_in) * 100.0:.2f}%)")
    print("-" * 80)
    print("REMOVAL ACCOUNTING BREAKDOWN:")
    print(f"  - Duplicate Doc IDs            : {purged_counts['duplicate_ids']:,}")
    print(f"  - Duplicate Content Hashes     : {purged_counts['duplicate_hashes']:,}")
    print(f"  - Reserved E-gen ID Collision  : {purged_counts['reserved_id_collisions']:,}")
    print(f"  - Reserved Hash Collision      : {purged_counts['reserved_hash_collisions']:,}")
    print(f"  - Narrow Tagore Keywords       : {purged_counts['narrow_keywords']:,}")
    print("  - 4-Work 5-Gram Contaminations :")
    for w_key, title in HELD_OUT_WORKS.items():
        print(f"      * {title:<18} : {purged_counts[w_key]:,}")
    print(f"  - Quality Rejects (<80% Bengali): {purged_counts['quality_rejected']:,}")
    print(f"  - Normalizer Invariant Violate : {purged_counts['invariant_violations']:,}")
    print("-" * 80)
    print("TOP 10 MOST FREQUENT NARROW TAGORE KEYWORDS TRIGGERED:")
    sorted_kws = sorted(keyword_frequency.items(), key=lambda kv: kv[1], reverse=True)[:10]
    for idx, (kw, cnt) in enumerate(sorted_kws, 1):
        pct = (cnt / docs_in) * 100.0 if docs_in > 0 else 0.0
        print(f"  {idx:>2}. {kw:<24} : {cnt:>6,} occurrences ({pct:.2f}% of stream)")
    print("-" * 80)
    print("10 EXAMPLE SNIPPETS OF PURGED KEYWORD DOCUMENTS:")
    for idx, item in enumerate(purged_snippets, 1):
        print(f"  [{idx:>2}] ID: {item['doc_id']} | Trigger: {item['keywords']}")
        print(f"       \"{item['snippet']}...\"")
    print("-" * 80)
    print("10M RUN REAL EPOCHS ACCOUNTING:")
    print(f"  Non-Embedding Params N_ne    : {n_ne_10m:,}")
    print(f"  100 * N_ne Bytes Required    : {target_bytes_10m:,} bytes ({target_bytes_10m / (1024**2):.2f} MB)")
    print(f"  Actual Cleaned Bytes Available: {bytes_out:,} bytes ({bytes_out / (1024**2):.2f} MB)")
    print(f"  Real Epochs Needed (10M run) : {real_epochs_10m:.3f} epochs")
    if bytes_out >= 1_000_000_000:
        print(f"  1.0 GB Volume Gate Status    : PASS ({bytes_out:,} >= 1,000,000,000 bytes)")
    else:
        diff = 1_000_000_000 - bytes_out
        print(f"  1.0 GB Volume Gate Status    : UNDER 1.0 GB ({diff:,} bytes remaining towards 1,000,000,000 bytes)")
    print("=" * 80)

    return {
        "docs_in": docs_in,
        "bytes_in": bytes_in,
        "docs_out": docs_out,
        "bytes_out": bytes_out,
        "total_purged_docs": total_purged_docs,
        "purged_counts": purged_counts,
        "keyword_frequency": dict(sorted_kws),
        "purged_snippets": purged_snippets,
        "plant_audit": plant_audit,
        "n_ne_10m": n_ne_10m,
        "target_bytes_10m": target_bytes_10m,
        "real_epochs_10m": real_epochs_10m,
        "output_file": str(output_p),
    }


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Post-hoc cleaning and verification of train.jsonl")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(r"G:\My Drive\RabindricGPT\data\raw\fineweb2\train.jsonl"),
        help="Input train.jsonl file",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(r"G:\My Drive\RabindricGPT\data\raw\fineweb2\train_clean.jsonl"),
        help="Cleaned output JSONL file",
    )
    parser.add_argument(
        "--pages-path",
        type=Path,
        default=Path(r"G:\My Drive\RabindricGPT\data\raw\wikisource\pages.jsonl"),
        help="Path to Wikisource pages.jsonl",
    )
    parser.add_argument(
        "--egen-path",
        type=Path,
        default=Path(r"G:\My Drive\RabindricGPT\data\raw\fineweb2\egen.jsonl"),
        help="Path to E-gen JSONL",
    )
    parser.add_argument(
        "--tok-sample-path",
        type=Path,
        default=Path(r"G:\My Drive\RabindricGPT\data\cleaned\fineweb2\fw2_tokenizer_sample_100m.txt"),
        help="Path to 100M tokenizer sample",
    )
    args = parser.parse_args()

    clean_train_file(
        input_file=args.input_file,
        output_file=args.output_file,
        pages_path=args.pages_path,
        egen_path=args.egen_path,
        tok_sample_path=args.tok_sample_path,
    )


if __name__ == "__main__":
    main()
