"""
scripts/audits/audit_flops.py : Audit analytical FLOPs per token and compute units across all 24 cells
(4 scales x 6 tokenizers) comparing planning doc 08 anchors with shard-measured sequence lengths.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rgpt.common.hashing import hash_file
from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import GPT


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def run_flops_audit() -> Dict[str, Any]:
    scales = ["10m", "25m", "50m", "100m"]
    tokenizers = ["byte", "char", "aks", "bpe8", "bpe16", "uni16"]

    input_shas = {}

    # 1. Non-embedding parameters from instantiated models
    models_info = {}
    for s in scales:
        p = Path(f"configs/models/{s}.yaml")
        input_shas[str(p)] = hash_file(p)
        m_cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        cfg = ModelConfig(
            size_id=s,
            vocab_size=16384,
            max_seq_len=m_cfg["max_seq_len"],
            n_layers=m_cfg["n_layers"],
            d_model=m_cfg["d_model"],
            n_heads=m_cfg["n_heads"],
            n_kv_heads=m_cfg["n_kv_heads"],
            head_dim=m_cfg["head_dim"],
            hidden_dim=m_cfg["hidden_dim"],
            rms_norm_eps=float(m_cfg["rms_norm_eps"]),
            rope_theta=float(m_cfg["rope_theta"]),
            tie_embeddings=m_cfg["tie_embeddings"],
        )
        m = GPT(cfg)
        total_p = sum(param.numel() for param in m.parameters())
        embed_p = m.w_token.weight.numel()
        n_ne = total_p - embed_p
        models_info[s] = {
            "L": m_cfg["n_layers"],
            "d": m_cfg["d_model"],
            "N_ne": n_ne,
        }

    # 2. Tokenizers info from configs and shard indices
    tok_info = {}
    for t in tokenizers:
        t_p = Path(f"configs/tokenizers/{t}.yaml")
        input_shas[str(t_p)] = hash_file(t_p)
        t_cfg = yaml.safe_load(t_p.read_text(encoding="utf-8"))

        idx_p = Path(f"shards/wikisource/{t}/train/index.json")
        input_shas[str(idx_p)] = hash_file(idx_p)
        meta = json.loads(idx_p.read_text(encoding="utf-8"))
        s_meas = meta["total_tokens"] / meta["total_chunks"]

        tok_info[t] = {
            "V": t_cfg["vocab_size"],
            "s_plan": t_cfg["max_window_tokens"],
            "s_meas": s_meas,
            "T_plan": t_cfg.get("bytes_per_token_plan", 1.0),
        }

    rows = []
    for s in scales:
        minfo = models_info[s]
        L = minfo["L"]
        d = minfo["d"]
        n_ne = minfo["N_ne"]
        d_bytes = 100 * n_ne

        for t in tokenizers:
            tinfo = tok_info[t]
            V = tinfo["V"]
            sp = tinfo["s_plan"]
            sm = tinfo["s_meas"]
            Tp = tinfo["T_plan"]

            # Analytical formula: 6*(N_ne + V*d) + 12*L*d*s
            f_plan = 6 * (n_ne + V * d) + 12 * L * d * sp
            f_meas = 6 * (n_ne + V * d) + 12 * L * d * sm

            # Compute units: D_tok planned vs measured
            dp = d_bytes / Tp
            dm = dp * (sm / sp)

            up = (f_plan * dp) / 2.2e16
            um = (f_meas * dm) / 2.2e16
            s_dev = (sm - sp) / sp

            rows.append(
                {
                    "scale": s,
                    "tokenizer": t,
                    "L": L,
                    "d": d,
                    "V": V,
                    "N_ne": n_ne,
                    "s_plan": sp,
                    "s_meas": sm,
                    "s_dev": s_dev,
                    "f_plan": f_plan,
                    "f_meas": f_meas,
                    "u_plan": up,
                    "u_meas": um,
                }
            )

    return {
        "git_sha": get_git_sha(),
        "input_shas": input_shas,
        "rows": rows,
    }


def main() -> None:
    res = run_flops_audit()
    out_dir = Path("results/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "audit_flops.txt"

    lines = []
    lines.append(
        "# Audit: 24-Cell Matrix FLOPs & Compute Accounting (Doc 08 Anchors vs Measured)"
    )
    lines.append(f"Git Commit SHA: {res['git_sha']}\n")
    lines.append("## Input File SHAs:")
    for path, sha in sorted(res["input_shas"].items()):
        lines.append(f"- {path}: {sha}")

    lines.append("\n## All 24 Cells: Architecture, FLOPs and Compute Units:")
    hdr = f"{'Scale':<5} | {'Tok':<6} | {'L':<2} | {'d':<4} | {'V':<6} | {'N_ne':<9} | {'s_plan':<6} | {'s_meas':<7} | {'s_dev%':<8} | {'F_plan (M)':<10} | {'F_meas (M)':<10} | {'U_plan':<7} | {'U_meas':<7}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in res["rows"]:
        lines.append(
            f"{r['scale']:<5} | {r['tokenizer']:<6} | {r['L']:<2} | {r['d']:<4} | {r['V']:<6} | "
            f"{r['N_ne']:<9} | {r['s_plan']:<6} | {r['s_meas']:<7.1f} | {r['s_dev']:+7.2%} | "
            f"{r['f_plan'] / 1e6:<10.2f} | {r['f_meas'] / 1e6:<10.2f} | {r['u_plan']:<7.3f} | {r['u_meas']:<7.3f}"
        )

    lines.append("\n## Sequence Length Drift Analysis (5% Gate):")
    for r in res["rows"][:6]:
        status = "PASS (<= 5%)" if abs(r["s_dev"]) <= 0.05 else "DRIFT (> 5%)"
        lines.append(
            f"- {r['tokenizer']:<6}: planned s = {r['s_plan']:<4}, measured s = {r['s_meas']:<6.1f} (dev: {r['s_dev']:+6.2%}) -> {status}"
        )

    report_str = "\n".join(lines) + "\n"
    print(report_str)
    out_txt.write_text(report_str, encoding="utf-8")
    print(f"\nPinned audit report written to {out_txt}")


if __name__ == "__main__":
    main()
