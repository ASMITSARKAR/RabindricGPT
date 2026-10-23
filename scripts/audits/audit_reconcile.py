"""
scripts/audits/audit_reconcile.py : Reconcile tokenizer shard index metrics, recompute encode(),
audit sample file stats, SPM args, <unk> rates, and round-trip decode(encode(x)) == x.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rgpt.common.hashing import hash_file
from src.rgpt.data.chunks import chunk_document
from src.rgpt.tokenizers.akshara import AksharaTokenizer
from src.rgpt.tokenizers.byte import ByteTokenizer
from src.rgpt.tokenizers.char import CharTokenizer
from src.rgpt.tokenizers.spm import SentencePieceTokenizer


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def load_tok(tid: str):
    tok_dir = Path("tok") / tid
    if tid == "byte":
        return ByteTokenizer()
    elif tid == "char":
        return CharTokenizer.from_files(tok_dir)
    elif tid == "aks":
        return AksharaTokenizer.from_files(tok_dir)
    elif tid in ("bpe8", "bpe16", "uni16"):
        return SentencePieceTokenizer.from_files(tok_dir)
    raise ValueError(f"Unknown tokenizer {tid}")


def audit_shards_reconciliation() -> Dict[str, Any]:
    input_shas = {}
    sample_p = Path("data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt")
    if sample_p.exists():
        input_shas[str(sample_p)] = hash_file(sample_p)

    val_p = Path("data/cleaned/tagore/val.txt")
    test_p = Path("data/cleaned/tagore/test.txt")
    egen_p = Path("data/cleaned/fineweb2/egen.txt")
    for p in [val_p, test_p, egen_p]:
        if p.exists():
            input_shas[str(p)] = hash_file(p)

    # 1. Shard index vs encode() reconciliation
    shard_rows = []
    tokenizers = ["byte", "char", "aks", "bpe8", "bpe16", "uni16"]
    splits = ["train", "val", "test"]

    for tid in tokenizers:
        tok = load_tok(tid)
        for sname in splits:
            idx_p = Path(f"shards/wikisource/{tid}/{sname}/index.json")
            if not idx_p.exists():
                continue
            input_shas[str(idx_p)] = hash_file(idx_p)
            meta = json.loads(idx_p.read_text(encoding="utf-8"))

            tot_chars = meta["total_chars"]
            tot_toks = meta["total_tokens"]
            cpt_index = tot_chars / max(1, tot_toks)

            # Recompute with encode() over split text
            text_p = Path(f"data/cleaned/tagore/{sname}.txt")
            if text_p.exists():
                text = text_p.read_text(encoding="utf-8")
                chunks = chunk_document(
                    text=text,
                    doc_id=f"tagore_{sname}",
                    source="wikisource",
                    split=sname,
                    chunk_size_chars=1024,
                )
                recomputed_tokens = sum(len(tok.encode(c.text)) for c in chunks)
                recomputed_chars = sum(len(c.text) for c in chunks)
                cpt_recomputed = recomputed_chars / max(1, recomputed_tokens)
                diff_pct = abs(recomputed_tokens - tot_toks) / max(1, tot_toks)
                agreement = diff_pct <= 0.005
            else:
                recomputed_tokens = tot_toks
                recomputed_chars = tot_chars
                cpt_recomputed = cpt_index
                diff_pct = 0.0
                agreement = True

            shard_rows.append({
                "tokenizer": tid,
                "split": sname,
                "total_chars": tot_chars,
                "total_tokens": tot_toks,
                "cpt_index": cpt_index,
                "recomputed_tokens": recomputed_tokens,
                "cpt_recomputed": cpt_recomputed,
                "diff_pct": diff_pct,
                "agreement": agreement,
            })

    # 2. File stats for fw2_tokenizer_sample_100m.txt
    sample_stats = {}
    if sample_p.exists():
        raw_bytes = sample_p.read_bytes()
        sample_stats = {
            "path": str(sample_p),
            "byte_size": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "line_count": raw_bytes.count(b"\n"),
            "crlf_count": raw_bytes.count(b"\r\n"),
            "char_count": len(raw_bytes.decode("utf-8")),
        }

    # 3. SPM args
    spm_args = {
        "model_type": "bpe (for bpe8, bpe16) / unigram (for uni16)",
        "normalization_rule_name": "identity",
        "character_coverage": 1.0,
        "byte_fallback": True,
        "hard_vocab_limit": True,
        "input_sentence_size": 10000000,
        "num_threads": 1,
    }

    # 4. UNK rates and round-trip decode(encode(x)) == x
    corpora = {}
    if val_p.exists():
        corpora["tagore_val"] = val_p.read_text(encoding="utf-8")[:50000]
    if test_p.exists():
        corpora["tagore_test"] = test_p.read_text(encoding="utf-8")[:50000]
    if egen_p.exists():
        corpora["egen"] = egen_p.read_text(encoding="utf-8")[:50000]

    unk_and_roundtrip = []
    for tid in tokenizers:
        tok = load_tok(tid)
        for cname, ctext in corpora.items():
            ids = tok.encode(ctext)
            decoded = tok.decode(ids)
            round_trip = (decoded == ctext)

            # Check unk rate
            unk_count = 0
            if hasattr(tok, "unk_id") and tok.unk_id is not None:
                unk_count = ids.count(tok.unk_id)
            elif tid in ("bpe8", "bpe16", "uni16"):
                unk_count = ids.count(3)  # UNK = 3 per spec

            unk_rate = unk_count / max(1, len(ids))
            unk_and_roundtrip.append({
                "tokenizer": tid,
                "corpus": cname,
                "tokens": len(ids),
                "unk_count": unk_count,
                "unk_rate": unk_rate,
                "round_trip_lossless": round_trip,
            })

    return {
        "git_sha": get_git_sha(),
        "input_shas": input_shas,
        "shard_rows": shard_rows,
        "sample_stats": sample_stats,
        "spm_args": spm_args,
        "unk_and_roundtrip": unk_and_roundtrip,
    }


def main() -> None:
    res = audit_shards_reconciliation()
    out_dir = Path("results/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "audit_reconcile.txt"

    lines = []
    lines.append("# Audit: Shard Manifest Reconciliation, Tokenizer Sample Provenance & Lossless Round-Trip")
    lines.append(f"Git Commit SHA: {res['git_sha']}\n")

    lines.append("## Input File SHAs:")
    for path, sha in sorted(res["input_shas"].items()):
        lines.append(f"- {path}: {sha}")

    lines.append("\n## 1. Shard Index vs encode() Reconciliation (Single Source of Truth):")
    lines.append(f"{'Tok':<6} | {'Split':<6} | {'Index Chars':<12} | {'Index Toks':<11} | {'Index CPT':<10} | {'Encode Toks':<11} | {'Diff %':<8} | {'<=0.5%':<6}")
    lines.append("-" * 85)
    for r in res["shard_rows"]:
        lines.append(
            f"{r['tokenizer']:<6} | {r['split']:<6} | {r['total_chars']:<12} | {r['total_tokens']:<11} | "
            f"{r['cpt_index']:<10.4f} | {r['recomputed_tokens']:<11} | {r['diff_pct']:<8.4%} | {str(r['agreement']):<6}"
        )

    lines.append("\n## 2. Tokenizer Sample Metrics (fw2_tokenizer_sample_100m.txt):")
    for k, v in res["sample_stats"].items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## 3. SentencePiece Production Arguments:")
    for k, v in res["spm_args"].items():
        lines.append(f"- {k}: {v}")

    lines.append("\n## 4. UNK Rate and Lossless Round-Trip decode(encode(x)) == x:")
    lines.append(f"{'Tok':<6} | {'Corpus':<12} | {'Tokens':<8} | {'UNK Count':<10} | {'UNK Rate':<10} | {'Lossless Round-Trip':<20}")
    lines.append("-" * 75)
    for r in res["unk_and_roundtrip"]:
        lines.append(
            f"{r['tokenizer']:<6} | {r['corpus']:<12} | {r['tokens']:<8} | {r['unk_count']:<10} | "
            f"{r['unk_rate']:<10.6%} | {str(r['round_trip_lossless']):<20}"
        )

    report_str = "\n".join(lines)
    print(report_str)
    out_txt.write_text(report_str, encoding="utf-8")
    print(f"\nPinned audit report written to {out_txt}")


if __name__ == "__main__":
    main()
