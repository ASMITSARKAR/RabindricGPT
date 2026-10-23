import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

from src.rgpt.common.hashing import hash_file


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def compute_pages_stats(pages_path: Path, config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        works_cfg = yaml.safe_load(f)["works"]

    pfx_to_work = {w["prefix"]: w for w in works_cfg}
    stats = defaultdict(lambda: {"count": 0, "clean_chars": 0})

    with open(pages_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pfx = row.get("prefix", "")
            if pfx in pfx_to_work:
                clean_text = row.get("clean_text") or ""
                stats[pfx]["count"] += 1
                stats[pfx]["clean_chars"] += len(clean_text)

    records = []
    for w in works_cfg:
        pfx = w["prefix"]
        st = stats[pfx]
        records.append({
            "id": w["id"],
            "genre": w["genre"],
            "pages": st["count"],
            "clean_chars": st["clean_chars"],
            "split": w["split"],
            "config_measured": w.get("clean_chars_measured", 0),
            "match": st["clean_chars"] == w.get("clean_chars_measured", 0),
        })

    input_shas = {
        str(pages_path): hash_file(pages_path),
        str(config_path): hash_file(config_path),
    }

    return {
        "git_sha": get_git_sha(),
        "input_shas": input_shas,
        "works": records,
    }


if __name__ == "__main__":
    pages_p = Path("data/raw/wikisource/pages.jsonl")
    cfg_p = Path("configs/works_wikisource.yaml")
    res = compute_pages_stats(pages_p, cfg_p)

    out_dir = Path("results/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "audit_splits.txt"

    lines = []
    lines.append("# Audit: Whole-Work Split Composition & Clean Character Verification")
    lines.append(f"Git Commit SHA: {res['git_sha']}\n")
    lines.append("## Input File SHAs:")
    for path, sha in sorted(res["input_shas"].items()):
        lines.append(f"- {path}: {sha}")
    lines.append("\n## Per-Work Measured Clean Characters from pages.jsonl:")
    lines.append(f"{'id':<16} | {'genre':<10} | {'pages':<6} | {'clean_chars':<12} | {'split':<6} | {'config_measured':<15} | {'match':<5}")
    lines.append("-" * 80)
    all_match = True
    for r in res["works"]:
        if not r["match"]:
            all_match = False
        lines.append(f"{r['id']:<16} | {r['genre']:<10} | {r['pages']:<6} | {r['clean_chars']:<12} | {r['split']:<6} | {r['config_measured']:<15} | {str(r['match']):<5}")

    lines.append(f"\nAll clean_chars_measured match pages.jsonl: {all_match}")

    report_str = "\n".join(lines)
    print(report_str)
    out_txt.write_text(report_str, encoding="utf-8")
    print(f"\nPinned audit report written to {out_txt}")
