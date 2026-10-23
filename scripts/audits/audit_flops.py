"""
scripts/audits/audit_flops.py : Audit analytical FLOPs per token and compute units across all 24 cells
(4 scales x 6 tokenizers) under the equal-bytes rule D_tok = D_bytes / T_meas for both FineWeb (Pretrain)
and Tagore (Adaptation) measured shards.
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
from src.rgpt.compute.flops import calculate_f_tok, calculate_run_flops, calculate_units


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

        idx_tagore = Path(f"shards/wikisource/{t}/train/index.json")
        input_shas[str(idx_tagore)] = hash_file(idx_tagore)
        meta_tagore = json.loads(idx_tagore.read_text(encoding="utf-8"))
        s_tagore = meta_tagore["total_tokens"] / meta_tagore["total_chunks"]

        idx_fw = Path(f"shards/fineweb2/{t}/train/index.json")
        input_shas[str(idx_fw)] = hash_file(idx_fw)
        meta_fw = json.loads(idx_fw.read_text(encoding="utf-8"))
        s_fw = meta_fw["total_tokens"] / meta_fw["total_chunks"]

        tok_info[t] = {
            "V": t_cfg["vocab_size"],
            "s_plan": t_cfg["max_window_tokens"],
            "s_tagore": s_tagore,
            "s_fw": s_fw,
        }

    tagore_bytes_per_win = 4237448 / 1573  # 2693.86
    fw_bytes_per_win = 2023293 / 749      # 2701.33

    pretrain_rows = []
    adapt_rows = []

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

            # FineWeb (Pretraining)
            sfw = tinfo["s_fw"]
            T_fw = fw_bytes_per_win / sfw
            d_tok_fw = d_bytes / T_fw
            f_tok_fw = 6 * (n_ne + V * d) + 12 * L * d * sfw
            c_fw = f_tok_fw * d_tok_fw
            u_fw = c_fw / 2.2e16
            s_dev_fw = (sfw - sp) / sp
            pretrain_rows.append({
                "scale": s, "tokenizer": t, "L": L, "d": d, "V": V, "N_ne": n_ne,
                "s_plan": sp, "s_meas": sfw, "s_dev": s_dev_fw, "T_meas": T_fw,
                "d_tok": d_tok_fw, "f_meas": f_tok_fw, "u_meas": u_fw,
            })

            # Tagore (Adaptation)
            stag = tinfo["s_tagore"]
            T_tag = tagore_bytes_per_win / stag
            d_tok_tag = d_bytes / T_tag
            f_tok_tag = 6 * (n_ne + V * d) + 12 * L * d * stag
            c_tag = f_tok_tag * d_tok_tag
            u_tag = c_tag / 2.2e16
            s_dev_tag = (stag - sp) / sp
            adapt_rows.append({
                "scale": s, "tokenizer": t, "L": L, "d": d, "V": V, "N_ne": n_ne,
                "s_plan": sp, "s_meas": stag, "s_dev": s_dev_tag, "T_meas": T_tag,
                "d_tok": d_tok_tag, "f_meas": f_tok_tag, "u_meas": u_tag,
            })

    return {
        "git_sha": get_git_sha(),
        "input_shas": input_shas,
        "pretrain_rows": pretrain_rows,
        "adapt_rows": adapt_rows,
    }


def main() -> None:
    res = run_flops_audit()
    out_dir = Path("results/audits")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / "audit_flops.txt"

    lines = []
    lines.append("# Audit: 24-Cell Matrix FLOPs & Compute Accounting (Equal-Bytes Rule: D_tok = D_bytes / T_meas)")
    lines.append(f"Git Commit SHA: {res['git_sha']}\n")
    lines.append("## Input File SHAs:")
    for path, sha in sorted(res["input_shas"].items()):
        lines.append(f"- {path}: {sha}")

    lines.append("\n## Note on Compute Unit Calibration Constant:")
    lines.append("- Constant: 2.2e16 FLOPs/unit is an unmeasured central planning prior (assumed ~50% MFU on A100/L4). Actual Colab GPU throughput run will calibrate empirical FLOPs/unit.")

    lines.append("\n## 1. Pretraining Cells (FineWeb Shards: 2,701.33 B/window):")
    hdr = f"{'Scale':<5} | {'Tok':<6} | {'L':<2} | {'d':<4} | {'V':<6} | {'N_ne':<9} | {'s_plan':<6} | {'s_meas':<7} | {'s_dev%':<8} | {'T (B/tok)':<9} | {'D_tok (M)':<10} | {'F_meas (M)':<10} | {'Units (u)':<9}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    fw_total_1seed = 0.0
    for r in res["pretrain_rows"]:
        fw_total_1seed += r["u_meas"]
        lines.append(
            f"{r['scale']:<5} | {r['tokenizer']:<6} | {r['L']:<2} | {r['d']:<4} | {r['V']:<6} | "
            f"{r['N_ne']:<9} | {r['s_plan']:<6} | {r['s_meas']:<7.1f} | {r['s_dev']:+7.2%} | "
            f"{r['T_meas']:<9.3f} | {r['d_tok'] / 1e6:<10.2f} | {r['f_meas'] / 1e6:<10.2f} | {r['u_meas']:<9.3f}"
        )
    lines.append(f"\nTotal Pretraining Units (1 seed across 24 cells): {fw_total_1seed:.2f} u")

    lines.append("\n## 2. Adaptation Cells (Tagore Shards: 2,693.86 B/window):")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    tag_total_1seed = 0.0
    for r in res["adapt_rows"]:
        tag_total_1seed += r["u_meas"]
        lines.append(
            f"{r['scale']:<5} | {r['tokenizer']:<6} | {r['L']:<2} | {r['d']:<4} | {r['V']:<6} | "
            f"{r['N_ne']:<9} | {r['s_plan']:<6} | {r['s_meas']:<7.1f} | {r['s_dev']:+7.2%} | "
            f"{r['T_meas']:<9.3f} | {r['d_tok'] / 1e6:<10.2f} | {r['f_meas'] / 1e6:<10.2f} | {r['u_meas']:<9.3f}"
        )
    lines.append(f"\nTotal Adaptation Units (1 seed across 24 cells): {tag_total_1seed:.2f} u")

    lines.append("\n## Sequence Length Drift Analysis (5% Gate on Tagore Shards):")
    for r in res["adapt_rows"][:6]:
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
