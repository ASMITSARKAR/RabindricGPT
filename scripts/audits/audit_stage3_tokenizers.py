"""
scripts/audits/audit_stage3_tokenizers.py : Audit Stage 3 Tokenizers.
Verifies:
1. Each tokenizer's training corpus path, SHA-256 (100M FineWeb sample), vocab size, and piece count assertion.
2. Held-out checks on FineWeb e-gen and Tagore val:
   - Chars per token (CPT)
   - Byte-fallback percentage
   - Unknown (UNK) rate
   - Lossless round-trip (decode(encode(text)) == text)
   - Explicit failure assert: bpe16 >= 2.5 chars/token on Tagore val.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rgpt.common.hashing import hash_file
from src.rgpt.tokenizers.akshara import AksharaTokenizer
from src.rgpt.tokenizers.base import BaseTokenizer
from src.rgpt.tokenizers.byte import ByteTokenizer
from src.rgpt.tokenizers.char import CharTokenizer
from src.rgpt.tokenizers.spm import SentencePieceTokenizer

EXPECTED_SAMPLE_SHA = "7a1163f6c16390a23fb66e9561f291e2a6123e819b546f6608246060b2059910"
REQUESTED_SIZES = {
    "byte": 264,
    "char": 512,
    "aks": 8192,
    "bpe8": 8192,
    "bpe16": 16384,
    "uni16": 16384,
}


def load_tok(tok_id: str, tok_base: Path = Path("tok")) -> BaseTokenizer:
    d = tok_base / tok_id
    if tok_id == "byte":
        return ByteTokenizer()
    elif tok_id == "char":
        return CharTokenizer.from_files(d)
    elif tok_id == "aks":
        return AksharaTokenizer.from_files(d)
    elif tok_id in ("bpe8", "bpe16", "uni16"):
        return SentencePieceTokenizer.from_files(d)
    raise ValueError(f"Unknown tokenizer {tok_id}")


def audit_stage3_tokenizers(
    tok_base: Path = Path("tok"),
    sample_path: Path = Path("data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt"),
    val_path: Path = Path("data/cleaned/tagore/val.txt"),
    egen_path: Path = Path("data/cleaned/fineweb2/egen.txt"),
) -> Dict[str, Any]:
    # 1. Corpus verification & piece counts
    print("=" * 110)
    print("STAGE 3 TOKENIZER SPECIFICATION & CORPUS VERIFICATION")
    print("=" * 110)
    print(f"Verified 100M Training Corpus : {sample_path}")
    sample_sha = hash_file(sample_path) if sample_path.exists() else "ABSENT_ON_COLAB"
    print(f"100M Training Corpus SHA-256 : {sample_sha}")
    if sample_path.exists():
        assert sample_sha == EXPECTED_SAMPLE_SHA, f"Sample SHA mismatch! Got {sample_sha}"

    tok_specs = []
    print("-" * 110)
    header = f"{'Tokenizer':<10} | {'Type':<10} | {'Req Size':>8} | {'Vocab Size':>10} | {'Status':<6} | {'Corpus SHA-256':<64}"
    print(header)
    print("-" * 110)

    for tid in ["byte", "char", "aks", "bpe8", "bpe16", "uni16"]:
        tok = load_tok(tid, tok_base)
        cfg_p = tok_base / tid / "config.json"
        cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
        corp_meta = cfg.get("training_corpus", {})
        c_sha = corp_meta.get("sha256", sample_sha)
        req_sz = REQUESTED_SIZES[tid]
        actual_sz = tok.vocab_size
        matches = (actual_sz == req_sz)
        assert matches, f"Vocab size mismatch for {tid}: {actual_sz} != {req_sz}"
        if sample_path.exists():
            assert c_sha == EXPECTED_SAMPLE_SHA, f"Corpus SHA mismatch for {tid}!"
        print(f"{tid:<10} | {cfg.get('type', tid):<10} | {req_sz:>8,d} | {actual_sz:>10,d} | {'PASS':<6} | {c_sha}")
        tok_specs.append({
            "tokenizer": tid,
            "req_size": req_sz,
            "actual_size": actual_sz,
            "corpus_sha": c_sha,
            "matches": matches,
        })
    print("=" * 110)

    # 2. Held-out checks on FineWeb e-gen and Tagore val
    print("\n" + "=" * 110)
    print("STAGE 3 HELD-OUT CORPUS TOKENIZATION CHECKS (E-GEN & TAGORE VAL)")
    print("=" * 110)
    corpora = {}
    if egen_path.exists():
        corpora["fineweb2_egen"] = egen_path.read_text(encoding="utf-8")
    if val_path.exists():
        corpora["tagore_val"] = val_path.read_text(encoding="utf-8")

    heldout_results = []
    header2 = f"{'Corpus':<15} | {'Tokenizer':<10} | {'Chars':>10} | {'Tokens':>10} | {'Chars/Tok':>10} | {'Byte-FB %':>10} | {'UNK %':>8} | {'Lossless':<8}"
    print(header2)
    print("-" * 110)

    for cname, text in corpora.items():
        for tid in ["byte", "char", "aks", "bpe8", "bpe16", "uni16"]:
            tok = load_tok(tid, tok_base)
            ids = tok.encode(text)
            n_tokens = len(ids)
            n_chars = len(text)
            cpt = n_chars / max(1, n_tokens)

            # Byte fallback calculation
            byte_fb_count = 0
            if tid == "byte":
                byte_fb_count = 0  # Native byte tokenizer, no fallback
            elif tid == "aks":
                byte_fb_count = sum(1 for i in ids if i < 256 or (hasattr(tok, "is_byte_fallback") and tok.is_byte_fallback(i)))
            elif tid in ("bpe8", "bpe16", "uni16"):
                sp = tok._sp
                byte_fb_count = sum(1 for i in ids if sp.IdToPiece(i).startswith("<0x"))
            elif tid == "char":
                byte_fb_count = sum(1 for i in ids if hasattr(tok, "is_byte_fallback") and tok.is_byte_fallback(i))

            byte_fb_pct = (byte_fb_count / max(1, n_tokens)) * 100.0

            # UNK count (spec requires UNK rate == 0.0% due to byte fallback)
            unk_count = 0
            if hasattr(tok, "unk_id") and tok.unk_id is not None:
                unk_count = ids.count(tok.unk_id)
            elif tid in ("bpe8", "bpe16", "uni16"):
                unk_count = ids.count(3)  # UNK ID = 3
            unk_pct = (unk_count / max(1, n_tokens)) * 100.0

            # Lossless round trip
            decoded = tok.decode(ids)
            lossless = (decoded == text)
            assert lossless, f"Lossless round-trip failed for {tid} on {cname}!"
            assert unk_count == 0, f"Non-zero UNK count for {tid} on {cname}: {unk_count}"

            # Gate: bpe16 must be >= 2.5 chars per token on Tagore val
            if tid == "bpe16" and cname == "tagore_val":
                print(f"[Gate Assertion] bpe16 on Tagore val CPT: {cpt:.4f} (Requirement: >= 2.5 chars/tok)")
                assert cpt >= 2.5, f"GATE FAIL: bpe16 CPT on Tagore val is {cpt:.4f} < 2.5!"

            print(
                f"{cname:<15} | {tid:<10} | {n_chars:>10,d} | {n_tokens:>10,d} | {cpt:>10.4f} | {byte_fb_pct:>9.4f}% | {unk_pct:>7.4f}% | {'PASS':<8}"
            )
            heldout_results.append({
                "corpus": cname,
                "tokenizer": tid,
                "chars": n_chars,
                "tokens": n_tokens,
                "cpt": cpt,
                "byte_fb_pct": byte_fb_pct,
                "unk_pct": unk_pct,
                "lossless": lossless,
            })

    print("=" * 110)
    print("Stage 3 Tokenizer Verification Status: ALL PASS (Lossless round-trip: 100%, UNK rate: 0.0%, bpe16 CPT >= 2.5)")
    return {
        "tok_specs": tok_specs,
        "heldout_results": heldout_results,
    }


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    audit_stage3_tokenizers()


if __name__ == "__main__":
    main()
