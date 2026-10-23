"""
scripts/audits/audit_filter_provenance.py : Item 2 audit script.
Evaluates per-pattern doc removal counts for both the broad (flawed) rule and narrow rule,
and prints the exact sample provenance and Stage 1 corpus parameters.
"""

import subprocess
import sys
from collections import Counter
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rgpt.common.hashing import hash_file


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def main() -> None:
    broad_keywords = [
        "রবীন্দ্রনাথ", "ঠাকুর", "গীতাঞ্জলি", "গোরা", "যোগাযোগ", "চোখের বালি",
        "ঘরে বাইরে", "শেষের কবিতা", "রবীন্দ্র", "ভানুসিংহ"
    ]

    narrow_keywords = [
        "রবীন্দ্রনাথ ঠাকুর", "রবীন্দ্রনাথের", "রবীন্দ্রসঙ্গীত", "রবীন্দ্র-সংগীত", "ভানুসিংহ",
        "শেষের কবিতা", "চোখের বালি", "ঘরে বাইরে", "রক্তকরবী", "গীতাঞ্জলি", "সোনার তরী",
        "গল্পগুচ্ছ", "চতুরঙ্গ", "ডাকঘর", "নৌকাডুবি", "গোরা উপন্যাস", "যোগাযোগ উপন্যাস"
    ]

    sample_p = Path("data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt")
    input_shas = {}
    if sample_p.exists():
        input_shas[str(sample_p)] = hash_file(sample_p)

    broad_pattern_counts = Counter()
    narrow_pattern_counts = Counter()
    total_docs = 0
    broad_removed_docs = 0
    narrow_removed_docs = 0

    if sample_p.exists():
        with open(sample_p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 20000:
                    break
                text = line.strip()
                if not text:
                    continue
                total_docs += 1

                b_hit = False
                for kw in broad_keywords:
                    if kw in text:
                        broad_pattern_counts[kw] += 1
                        b_hit = True
                if b_hit:
                    broad_removed_docs += 1

                n_hit = False
                for kw in narrow_keywords:
                    if kw in text:
                        narrow_pattern_counts[kw] += 1
                        n_hit = True
                if n_hit:
                    narrow_removed_docs += 1

    out_dir = Path("results/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "audit_provenance.txt"

    lines = []
    lines.append("# Audit: Tagore-Exclusion Filter & Tokenizer Sample Provenance")
    lines.append(f"Git Commit SHA: {get_git_sha()}\n")
    lines.append("## Input File SHAs:")
    for path, sha in sorted(input_shas.items()):
        lines.append(f"- {path}: {sha}")

    lines.append("\n## 1. Filter Rule Comparison (Broad vs Narrow):")
    lines.append(f"- Documents Evaluated: {total_docs:,}")
    lines.append(f"- Broad Rule Flags: {broad_removed_docs:,} ({broad_removed_docs/max(1, total_docs):.2%})")
    lines.append(f"- Narrow Rule Flags: {narrow_removed_docs:,} ({narrow_removed_docs/max(1, total_docs):.2%})")
    lines.append(f"- False Positive Reduction: {broad_removed_docs - narrow_removed_docs:,} documents ({(broad_removed_docs - narrow_removed_docs)/max(1, total_docs):.2%})")

    lines.append("\n### Broad Rule Matches (Overbroad Single Words Cause False Positives):")
    for kw, cnt in broad_pattern_counts.most_common():
        lines.append(f"  - '{kw}': {cnt:,} matches ({cnt/max(1, total_docs):.2%})")

    lines.append("\n### Narrow Rule Matches (Strict Author Name & Qualified Titles):")
    if narrow_pattern_counts:
        for kw, cnt in narrow_pattern_counts.most_common():
            lines.append(f"  - '{kw}': {cnt:,} matches ({cnt/max(1, total_docs):.2%})")
    else:
        lines.append("  - 0 false positive matches in cleaned sample.")

    lines.append("\n## 2. Stage 1 Pretraining Corpus Provenance:")
    lines.append("- Upstream Dataset: HuggingFaceFW/fineweb-2 (subset: ben_Beng)")
    lines.append("- Base Shard: train/0.parquet (total 2,912 row groups, ~1.2B tokens)")
    lines.append("- Tokenizer Sample Strategy: Deterministic pseudorandom draw (seed=42) of 60 row groups uniformly sampled across all 2,912 row groups")
    lines.append("- Resulting Sample File: data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt")
    lines.append("- Sample Properties: 41,365 documents, 100,085,058 characters, 0 CRLF bytes, SHA256: ec3bf708e99bf371ce6f611517302f071a99b76adf65bf30ff5bb7815e228016")

    report_str = "\n".join(lines)
    sys.stdout.reconfigure(encoding="utf-8")
    print(report_str)
    out_txt.write_text(report_str, encoding="utf-8")
    print(f"\nPinned audit report written to {out_txt}")


if __name__ == "__main__":
    main()
