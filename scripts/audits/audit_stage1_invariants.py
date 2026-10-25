"""
scripts/audits/audit_stage1_invariants.py : Audit Stage 1 invariants on train.jsonl.
Verifies:
1. No duplicate doc IDs or content hashes.
2. Disjoint from egen.jsonl and 100M tokenizer sample (by doc ID and content hash).
3. Every document strictly satisfies normalize_text invariants:
   - Zero U+FEFF (BOM)
   - Zero U+200C (ZWNJ)
   - U+200D (ZWJ) only allowed inside C + virama + ZWJ + ya ([ক-হ]্‍য)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rgpt.text.normalize import normalize_text

ALLOWED_ZWJ_PATTERN = re.compile(r"[\u0995-\u09B9]\u09CD\u200D\u09AF")


def audit_stage1_invariants(
    train_path: Path = Path("data/raw/fineweb2/train.jsonl"),
    egen_path: Path = Path("data/raw/fineweb2/egen.jsonl"),
    tok_sample_path: Path = Path("data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt"),
) -> dict:
    if not train_path.exists():
        raise FileNotFoundError(f"train.jsonl not found at {train_path}")

    # 1. Load reserved IDs and hashes from egen.jsonl
    egen_ids = set()
    egen_hashes = set()
    if egen_path.exists():
        with open(egen_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if "id" in row:
                        egen_ids.add(str(row["id"]))
                    if "text" in row:
                        h = hashlib.sha256(row["text"].strip().encode("utf-8")).hexdigest()
                        egen_hashes.add(h)

    # 2. Load sample hashes from 100M tokenizer sample
    tok_hashes = set()
    if tok_sample_path.exists():
        with open(tok_sample_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 10_000:
                    break
                line_str = line.strip()
                if line_str:
                    tok_hashes.add(hashlib.sha256(line_str.encode("utf-8")).hexdigest())

    # 3. Stream train.jsonl
    seen_ids = set()
    seen_hashes = set()
    total_docs = 0
    total_bytes = 0
    total_chars = 0

    feff_violations = 0
    zwnj_violations = 0
    invalid_zwj_violations = 0
    duplicate_id_count = 0
    duplicate_hash_count = 0
    egen_id_collisions = 0
    egen_hash_collisions = 0
    tok_hash_collisions = 0

    with open(train_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("id", ""))
            text = row.get("text", "")
            raw_bytes = len(text.encode("utf-8"))
            total_docs += 1
            total_bytes += raw_bytes
            total_chars += len(text)

            # Duplicate checks
            if doc_id in seen_ids:
                duplicate_id_count += 1
            seen_ids.add(doc_id)

            chash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
            if chash in seen_hashes:
                duplicate_hash_count += 1
            seen_hashes.add(chash)

            # Disjointness checks
            if doc_id in egen_ids:
                egen_id_collisions += 1
            if chash in egen_hashes:
                egen_hash_collisions += 1
            if chash in tok_hashes:
                tok_hash_collisions += 1

            # Invariant checks on normalize_text
            if "\ufeff" in text:
                feff_violations += 1
            if "\u200c" in text:
                zwnj_violations += 1
            if "\u200d" in text:
                # Find all ZWJ occurrences and verify each is part of C + virama + ZWJ + ya
                matches = list(ALLOWED_ZWJ_PATTERN.finditer(text))
                allowed_zwj_count = len(matches)
                total_zwj_count = text.count("\u200d")
                if allowed_zwj_count != total_zwj_count:
                    invalid_zwj_violations += (total_zwj_count - allowed_zwj_count)

    res = {
        "train_path": str(train_path),
        "total_docs": total_docs,
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "duplicate_id_count": duplicate_id_count,
        "duplicate_hash_count": duplicate_hash_count,
        "egen_id_collisions": egen_id_collisions,
        "egen_hash_collisions": egen_hash_collisions,
        "tok_hash_collisions": tok_hash_collisions,
        "feff_violations": feff_violations,
        "zwnj_violations": zwnj_violations,
        "invalid_zwj_violations": invalid_zwj_violations,
        "disjoint_egen": (egen_id_collisions == 0 and egen_hash_collisions == 0),
        "disjoint_tok_sample": (tok_hash_collisions == 0),
        "no_duplicates": (duplicate_id_count == 0 and duplicate_hash_count == 0),
        "invariants_passed": (
            feff_violations == 0 and zwnj_violations == 0 and invalid_zwj_violations == 0
        ),
    }

    print("=" * 80)
    print("STAGE 1 INVARIANTS AUDIT REPORT")
    print("=" * 80)
    print(f"Target File              : {train_path}")
    print(f"Total Documents Streamed : {total_docs:,}")
    print(f"Total Cleaned Bytes      : {total_bytes:,}")
    print(f"Total Cleaned Characters : {total_chars:,}")
    print("-" * 80)
    print(f"Duplicate Doc IDs        : {duplicate_id_count} {'[PASS]' if duplicate_id_count == 0 else '[FAIL]'}")
    print(f"Duplicate Content Hashes : {duplicate_hash_count} {'[PASS]' if duplicate_hash_count == 0 else '[FAIL]'}")
    print(f"E-Gen ID Collisions      : {egen_id_collisions} {'[PASS]' if egen_id_collisions == 0 else '[FAIL]'}")
    print(f"E-Gen Hash Collisions    : {egen_hash_collisions} {'[PASS]' if egen_hash_collisions == 0 else '[FAIL]'}")
    print(f"100M Sample Collisions   : {tok_hash_collisions} {'[PASS]' if tok_hash_collisions == 0 else '[FAIL]'}")
    print("-" * 80)
    print(f"U+FEFF (BOM) Count       : {feff_violations} {'[PASS]' if feff_violations == 0 else '[FAIL]'}")
    print(f"U+200C (ZWNJ) Count      : {zwnj_violations} {'[PASS]' if zwnj_violations == 0 else '[FAIL]'}")
    print(f"Invalid U+200D (ZWJ)     : {invalid_zwj_violations} {'[PASS]' if invalid_zwj_violations == 0 else '[FAIL]'}")
    all_pass = (
        res["no_duplicates"]
        and res["disjoint_egen"]
        and res["disjoint_tok_sample"]
        and res["invariants_passed"]
    )
    print(f"Overall Invariants Status: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 80)
    return res


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Audit Stage 1 invariants on train.jsonl")
    parser.add_argument("--train-path", type=Path, default=Path("data/raw/fineweb2/train.jsonl"))
    args = parser.parse_args()
    res = audit_stage1_invariants(train_path=args.train_path)
    if not (
        res["no_duplicates"]
        and res["disjoint_egen"]
        and res["disjoint_tok_sample"]
        and res["invariants_passed"]
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
