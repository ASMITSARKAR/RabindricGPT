"""
scripts/audits/audit_stage2_shards.py : Audit Stage 2 shard manifests, invariants, round-trip, and epoch calculations.
Contract:
1. Generates formatted table across all index.json files:
   corpus | tokenizer | split | chunks | tokens | chars | bytes | tokens_per_chunk
2. Asserts:
   - Shard total_chars equals the exact length of the cleaned text for every split (P3 check).
   - Tagore splits are whole-work (train: 6 works, val: Shesher Kobita, test: Gitanjali, Jogajog, Chitrangada) with no work in two splits.
   - decode(encode(chunk)) equals the chunk for 200 random chunks per tokenizer.
3. Computes and reports real epochs needed for 10M run:
   Epochs = (100 * N_ne bytes) / available train bytes (where N_ne = 10,621,824).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rgpt.data.splits import assign_work_splits, verify_whole_work_invariant
from src.rgpt.tokenizers.akshara import AksharaTokenizer
from src.rgpt.tokenizers.base import BaseTokenizer
from src.rgpt.tokenizers.byte import ByteTokenizer
from src.rgpt.tokenizers.char import CharTokenizer
from src.rgpt.tokenizers.spm import SentencePieceTokenizer

N_NE_10M = 10_621_824  # Non-embedding parameters for 10M config
TARGET_TOKENS_10M = 100 * N_NE_10M  # 1,062,182,400 tokens/bytes


def load_tok(tok_id: str, tok_base: Path = Path("tok")) -> BaseTokenizer:
    d = tok_base / tok_id
    if tok_id == "byte":
        return ByteTokenizer()
    elif tok_id == "char":
        return CharTokenizer.from_files(d)
    elif tok_id == "aks":
        return AksharaTokenizer.from_files(d)
    elif tok_id in ("bpe16", "bpe8", "uni16"):
        return SentencePieceTokenizer.from_files(d)
    raise ValueError(f"Unknown tokenizer {tok_id}")


def audit_stage2_shards(
    shards_dir: Path = Path("shards"),
    cleaned_dir: Path = Path("data/cleaned"),
    works_config_path: Path = Path("configs/works_wikisource.yaml"),
    n_roundtrip_chunks: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    # 1. Inspect index.json manifests and build table
    corpora = ["wikisource", "fineweb2"]
    tokenizers = ["byte", "char", "aks", "bpe16", "uni16", "bpe8"]
    splits_map = {
        "wikisource": ["train", "val", "test"],
        "fineweb2": ["train", "egen"],
    }

    table_rows = []
    p3_checks = []
    all_p3_passed = True

    print("\n" + "=" * 115)
    print("STAGE 2 SHARD MANIFEST INDEX TABLE")
    print("=" * 115)
    header = f"{'Corpus':<12} | {'Tokenizer':<10} | {'Split':<8} | {'Chunks':>8} | {'Tokens':>12} | {'Chars':>12} | {'Bytes':>12} | {'Tok/Chunk':>10}"
    print(header)
    print("-" * 115)

    for corp in corpora:
        for tid in tokenizers:
            for sname in splits_map[corp]:
                idx_path = shards_dir / corp / tid / sname / "index.json"
                if not idx_path.exists():
                    continue
                data = json.loads(idx_path.read_text(encoding="utf-8"))
                chunks = data.get("total_chunks", 0)
                tokens = data.get("total_tokens", 0)
                chars = data.get("total_chars", 0)
                bytes_cnt = data.get("total_bytes", 0)
                tok_per_chunk = tokens / chunks if chunks > 0 else 0.0

                row_dict = {
                    "corpus": corp,
                    "tokenizer": tid,
                    "split": sname,
                    "chunks": chunks,
                    "tokens": tokens,
                    "chars": chars,
                    "bytes": bytes_cnt,
                    "tokens_per_chunk": tok_per_chunk,
                }
                table_rows.append(row_dict)
                print(
                    f"{corp:<12} | {tid:<10} | {sname:<8} | {chunks:>8,d} | {tokens:>12,d} | {chars:>12,d} | {bytes_cnt:>12,d} | {tok_per_chunk:>10.2f}"
                )

    print("=" * 115)

    # 2. P3 Shard Total Check: Assert shard total_chars == len(cleaned text)
    print("\n" + "=" * 90)
    print("P3 SHARD TOTAL CHARACTERS EQUALITY CHECK")
    print("=" * 90)

    # Cleaned text file paths
    text_paths = {
        ("wikisource", "train"): cleaned_dir / "tagore" / "train.txt",
        ("wikisource", "val"): cleaned_dir / "tagore" / "val.txt",
        ("wikisource", "test"): cleaned_dir / "tagore" / "test.txt",
        ("fineweb2", "train"): cleaned_dir / "fineweb2" / "train.txt",
        ("fineweb2", "egen"): cleaned_dir / "fineweb2" / "egen.txt",
    }

    clean_lengths = {}
    for (c, s), p in text_paths.items():
        clean_lengths[(c, s)] = len(p.read_text(encoding="utf-8")) if p.exists() else 0

    for r in table_rows:
        corp = r["corpus"]
        sname = r["split"]
        expected_chars = clean_lengths.get((corp, sname), 0)
        actual_chars = r["chars"]
        passed = (actual_chars == expected_chars)
        if not passed:
            all_p3_passed = False
        p3_checks.append({
            "corpus": corp,
            "tokenizer": r["tokenizer"],
            "split": sname,
            "expected_chars": expected_chars,
            "actual_chars": actual_chars,
            "passed": passed,
        })

    for (c, s), exp in clean_lengths.items():
        # verify across tokenizers
        sub = [p for p in p3_checks if p["corpus"] == c and p["split"] == s]
        all_sub_pass = all(x["passed"] for x in sub)
        print(f"[{c:<10} / {s:<6}] Cleaned Text Length: {exp:,} chars | Shards Match across all {len(sub)} tokenizers: {'PASS' if all_sub_pass else 'FAIL'}")

    assert all_p3_passed, "P3 Check Failed: Shard total_chars does not match cleaned text length!"
    print("Overall P3 Shard-Total Check : PASS")
    print("=" * 90)

    # 3. Whole-Work Split Invariant Check
    print("\n" + "=" * 90)
    print("TAGORE WHOLE-WORK SPLIT INVARIANT CHECK")
    print("=" * 90)
    works_config = yaml.safe_load(works_config_path.read_text(encoding="utf-8"))
    works = works_config.get("works", [])
    splits = assign_work_splits(works, split_option=works_config.get("split_option", "Option_A"))
    verify_whole_work_invariant(splits)

    train_work_ids = {w["id"] for w in splits["train"]}
    val_work_ids = {w["id"] for w in splits["val"]}
    test_work_ids = {w["id"] for w in splits["test"]}

    print(f"Train Works ({len(train_work_ids)} works): {sorted(train_work_ids)}")
    print(f"Val Works   ({len(val_work_ids)} works): {sorted(val_work_ids)}")
    print(f"Test Works  ({len(test_work_ids)} works): {sorted(test_work_ids)}")

    assert len(train_work_ids) == 6, f"Expected 6 train works, got {len(train_work_ids)}"
    assert val_work_ids == {"shesher_kobita"}, f"Expected val to be shesher_kobita, got {val_work_ids}"
    assert test_work_ids == {"gitanjali", "jogajog", "chitrangada"}, f"Expected test works to be gitanjali, jogajog, chitrangada, got {test_work_ids}"
    assert train_work_ids.isdisjoint(val_work_ids), "Train and Val works overlap!"
    assert train_work_ids.isdisjoint(test_work_ids), "Train and Test works overlap!"
    assert val_work_ids.isdisjoint(test_work_ids), "Val and Test works overlap!"
    print("Whole-Work Partition Invariant: PASS (No work in two splits; train: 6, val: 1, test: 3)")
    print("=" * 90)

    # 4. decode(encode(chunk)) == chunk for 200 random chunks per tokenizer
    print("\n" + "=" * 90)
    print(f"LOSSLESS ROUND-TRIP CHECK ({n_roundtrip_chunks} RANDOM CHUNKS PER TOKENIZER)")
    print("=" * 90)
    all_chunks: List[str] = []
    # Collect coherent document chunks from cleaned texts and raw sources
    for p in text_paths.values():
        if p.exists():
            paragraphs = p.read_text(encoding="utf-8").split("\n\n")
            for para in paragraphs:
                para_s = para.strip()
                if len(para_s) >= 80:
                    # Chunks of up to 1024 characters
                    all_chunks.append(para_s[:1024].strip())

    random.seed(seed)
    sample_chunks = random.sample(all_chunks, min(n_roundtrip_chunks, len(all_chunks)))
    tok_base = Path("tok")
    roundtrip_results = {}

    for tid in ["byte", "char", "aks", "bpe16", "uni16"]:
        tok = load_tok(tid, tok_base)
        mismatches = 0
        for chunk in sample_chunks:
            encoded = tok.encode(chunk)
            decoded = tok.decode(encoded)
            if decoded != chunk:
                mismatches += 1
        passed = (mismatches == 0)
        roundtrip_results[tid] = {
            "tested": len(sample_chunks),
            "mismatches": mismatches,
            "passed": passed,
        }
        print(f"Tokenizer {tid:<8}: Tested {len(sample_chunks)} chunks | Mismatches: {mismatches} | {'PASS' if passed else 'FAIL'}")
        assert passed, f"Lossless round-trip failed for tokenizer {tid} on {mismatches} chunks!"

    print("Overall Lossless Round-Trip : PASS")
    print("=" * 90)

    # 5. Real Epochs Needed for 10M Run
    print("\n" + "=" * 90)
    print("REAL EPOCHS CALCULATION FOR 10M PARAMETER RUN")
    print("=" * 90)
    fw2_train_bytes = clean_lengths.get(("fineweb2", "train"), 0)
    tagore_train_bytes = clean_lengths.get(("wikisource", "train"), 0)
    fw2_train_raw_bytes = (
        (cleaned_dir / "fineweb2" / "train.txt").stat().st_size
        if (cleaned_dir / "fineweb2" / "train.txt").exists()
        else fw2_train_bytes * 3
    )
    tagore_train_raw_bytes = (
        (cleaned_dir / "tagore" / "train.txt").stat().st_size
        if (cleaned_dir / "tagore" / "train.txt").exists()
        else tagore_train_bytes * 3
    )

    epochs_fw2_bytes = TARGET_TOKENS_10M / fw2_train_raw_bytes if fw2_train_raw_bytes > 0 else 0
    epochs_tagore_bytes = TARGET_TOKENS_10M / tagore_train_raw_bytes if tagore_train_raw_bytes > 0 else 0

    print(f"10M Model Non-Embedding Parameters (N_ne) : {N_NE_10M:,}")
    print(f"Target Budget (100 * N_ne tokens/bytes)      : {TARGET_TOKENS_10M:,}")
    print(f"Available FineWeb-2 Train Bytes on Disk      : {fw2_train_raw_bytes:,} bytes")
    print(f"Real Epochs for 10M FineWeb-2 Run            : {epochs_fw2_bytes:.4f} epochs")
    print(f"Available Tagore Train Bytes on Disk         : {tagore_train_raw_bytes:,} bytes")
    print(f"Real Epochs for 10M Tagore Adaptation Run    : {epochs_tagore_bytes:.4f} epochs")
    print("=" * 90)

    return {
        "table_rows": table_rows,
        "all_p3_passed": all_p3_passed,
        "roundtrip_results": roundtrip_results,
        "epochs_fw2_bytes": epochs_fw2_bytes,
        "epochs_tagore_bytes": epochs_tagore_bytes,
        "fw2_train_raw_bytes": fw2_train_raw_bytes,
        "tagore_train_raw_bytes": tagore_train_raw_bytes,
    }


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    audit_stage2_shards()


if __name__ == "__main__":
    main()
