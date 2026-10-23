"""
scripts/audits/audit_flops.py : Audit empirical FLOPs per token across all scales and tokenizers
against doc 08 hand anchors.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

from src.rgpt.common.hashing import hash_file
from src.rgpt.compute.flops import calculate_f_tok


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_flops_audit() -> Dict[str, Any]:
    anchors = {
        "10m": {
            "char": calculate_f_tok(10616832, 512, 384, 6, 1024),
            "aks": calculate_f_tok(10616832, 8192, 384, 6, 640),
            "bpe16": calculate_f_tok(10616832, 16384, 384, 6, 256),
            "uni16": calculate_f_tok(10616832, 16384, 384, 6, 256),
            "bpe8": calculate_f_tok(10616832, 8192, 384, 6, 320),
        },
        "25m": {
            "char": calculate_f_tok(25165824, 512, 512, 8, 1024),
            "aks": calculate_f_tok(25165824, 8192, 512, 8, 640),
            "bpe16": calculate_f_tok(25165824, 16384, 512, 8, 256),
            "uni16": calculate_f_tok(25165824, 16384, 512, 8, 256),
            "bpe8": calculate_f_tok(25165824, 8192, 512, 8, 320),
        },
        "50m": {
            "char": calculate_f_tok(49152000, 512, 640, 10, 1024),
            "aks": calculate_f_tok(49152000, 8192, 640, 10, 640),
            "bpe16": calculate_f_tok(49152000, 16384, 640, 10, 256),
            "uni16": calculate_f_tok(49152000, 16384, 640, 10, 256),
            "bpe8": calculate_f_tok(49152000, 8192, 640, 10, 320),
        },
        "100m": {
            "char": calculate_f_tok(99090432, 512, 768, 14, 1024),
            "aks": calculate_f_tok(99090432, 8192, 768, 14, 640),
            "bpe16": calculate_f_tok(99090432, 16384, 768, 14, 256),
            "uni16": calculate_f_tok(99090432, 16384, 768, 14, 256),
            "bpe8": calculate_f_tok(99090432, 8192, 768, 14, 320),
        },
    }

    input_shas = {}
    for scale in ["10m", "25m", "50m", "100m"]:
        p = Path(f"configs/models/{scale}.yaml")
        input_shas[str(p)] = hash_file(p)

    rows = []
    max_dev = 0.0

    for scale in ["10m", "25m", "50m", "100m"]:
        m_cfg = yaml.safe_load(Path(f"configs/models/{scale}.yaml").read_text(encoding="utf-8"))
        L = m_cfg["n_layers"]
        d = m_cfg["d_model"]
        n_ne = 12 * L * (d**2)

        for tid in ["char", "aks", "bpe8", "bpe16", "uni16"]:
            idx_p = Path(f"shards/wikisource/{tid}/train/index.json")
            if not idx_p.exists():
                continue
            input_shas[str(idx_p)] = hash_file(idx_p)

            tok_p = Path(f"configs/tokenizers/{tid}.yaml")
            input_shas[str(tok_p)] = hash_file(tok_p)
            tok_cfg = yaml.safe_load(tok_p.read_text(encoding="utf-8"))
            V = tok_cfg["vocab_size"]

            meta = json.loads(idx_p.read_text(encoding="utf-8"))
            s_emp = int(meta["total_tokens"] // meta["total_chunks"])
            f_emp = calculate_f_tok(n_ne=n_ne, vocab_size=V, d_model=d, n_layers=L, s_tokens=s_emp)
            target = anchors[scale][tid]
            dev = abs(f_emp - target) / target
            if dev > max_dev:
                max_dev = dev

            rows.append({
                "scale": scale,
                "tokenizer": tid,
                "L": L,
                "d": d,
                "V": V,
                "s_emp": s_emp,
                "f_emp": f_emp,
                "target": target,
                "dev": dev,
            })

    return {
        "git_sha": get_git_sha(),
        "input_shas": input_shas,
        "rows": rows,
        "max_dev": max_dev,
    }


def main() -> None:
    res = run_flops_audit()
    out_dir = Path("results/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "audit_flops.txt"

    lines = []
    lines.append("# Audit: FLOPs Formula Verification vs Budget Doc 08 Anchors")
    lines.append(f"Git Commit SHA: {res['git_sha']}\n")
    lines.append("## Input File SHAs:")
    for path, sha in sorted(res["input_shas"].items()):
        lines.append(f"- {path}: {sha}")
    lines.append("\n## Empirical FLOPs vs Budget Doc Anchors:")
    lines.append(f"{'Scale':<6} | {'Tok':<6} | {'L':<3} | {'d':<4} | {'V':<6} | {'s_emp':<6} | {'F_emp':<10} | {'Budget Doc':<10} | {'Dev %':<8}")
    lines.append("-" * 75)
    for r in res["rows"]:
        lines.append(
            f"{r['scale']:<6} | {r['tokenizer']:<6} | {r['L']:<3} | {r['d']:<4} | {r['V']:<6} | "
            f"{r['s_emp']:<6} | {r['f_emp']/1e6:7.2f}M | {r['target']/1e6:7.2f}M | {r['dev']:6.4%}"
        )
    lines.append(f"\nMax deviation across all models and tokenizers: {res['max_dev']:.4%}")
    lines.append(f"Assertion <= 1.0%: {'PASSED' if res['max_dev'] < 0.01 else 'FAILED'}")

    report_str = "\n".join(lines)
    print(report_str)
    out_txt.write_text(report_str, encoding="utf-8")
    print(f"\nPinned audit report written to {out_txt}")


if __name__ == "__main__":
    main()
