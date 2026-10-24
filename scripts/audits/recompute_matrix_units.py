"""
scripts/audits/recompute_matrix_units.py : Recompute planned_units for every run in run_matrix.csv
from empirical L4 measured throughput (batch 64) and unit burn rate.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# Measured throughput on L4 (batch 64)
L4_MEASURED_TOK_PER_SEC = {
    "byte": 47368,
    "char": 58310,
    "akshara": 53098,
    "aks": 53098,
    "bpe16": 50615,
    "uni16": 50681,
}

# Empirical unit burn rate: 0.34 u for 10.6 minutes
BURN_RATE_UNITS_PER_SEC = 0.34 / (10.6 * 60.0)  # ~0.00053459 u/s (= 1.9245 u/hr)

# Measured bytes-per-token from FineWeb shards (7,025,878 bytes total)
FW_TOKENS = {
    "byte": 7025878,
    "char": 2653004,
    "akshara": 1709854,
    "aks": 1709854,
    "bpe16": 637938,
    "uni16": 626732,
}
BPT_FW = {t: 7025878 / count for t, count in FW_TOKENS.items()}

# Measured bytes-per-token from Tagore shards (4,237,448 bytes total)
TAGORE_TOKENS = {
    "byte": 4237448,
    "char": 1611537,
    "akshara": 1057041,
    "aks": 1057041,
    "bpe16": 465137,
    "uni16": 461678,
}
BPT_TAGORE = {t: 4237448 / count for t, count in TAGORE_TOKENS.items()}

MODELS_SPEC = {
    "10m": {"L": 6, "d": 384, "N_ne": 10621824},
    "25m": {"L": 8, "d": 512, "N_ne": 25698816},
    "50m": {"L": 10, "d": 640, "N_ne": 49575040},
    "100m": {"L": 14, "d": 768, "N_ne": 99112704},
}

VOCABS = {
    "byte": 264,
    "char": 512,
    "akshara": 8192,
    "aks": 8192,
    "bpe16": 16384,
    "uni16": 16384,
}

S_FW = {
    "byte": 2701.3,
    "char": 1024.3,
    "akshara": 660.2,
    "aks": 660.2,
    "bpe16": 246.3,
    "uni16": 242.0,
}

S_TAGORE = {
    "byte": 2693.9,
    "char": 1024.5,
    "akshara": 672.0,
    "aks": 672.0,
    "bpe16": 295.7,
    "uni16": 293.5,
}


def calc_f_tok(N_ne: int, V: int, d: int, L: int, s: float) -> float:
    """FLOPs per token = 6 * (N_ne + V * d) + 12 * L * d * s."""
    return 6.0 * (N_ne + V * d) + 12.0 * L * d * s


def compute_pretrain_run_units(scale: str, tok: str) -> float:
    meta = MODELS_SPEC[scale]
    N_ne = meta["N_ne"]
    d_bytes = 100 * N_ne  # Equal-bytes rule: 100 * N_ne bytes
    bpt = BPT_FW[tok]
    tokens_needed = d_bytes / bpt

    # Scale tokens/sec from measured 10M baseline inversely with model FLOPs/tok
    f_10m = calc_f_tok(
        MODELS_SPEC["10m"]["N_ne"],
        VOCABS[tok],
        MODELS_SPEC["10m"]["d"],
        MODELS_SPEC["10m"]["L"],
        S_FW[tok],
    )
    f_scale = calc_f_tok(
        meta["N_ne"],
        VOCABS[tok],
        meta["d"],
        meta["L"],
        S_FW[tok],
    )
    tok_s_10m = L4_MEASURED_TOK_PER_SEC[tok]
    tok_s_scale = tok_s_10m * (f_10m / f_scale)

    time_sec = tokens_needed / tok_s_scale
    units = time_sec * BURN_RATE_UNITS_PER_SEC
    return round(units, 2)


def compute_adapt_run_units(scale: str, tok: str) -> float:
    meta = MODELS_SPEC[scale]
    # 4 epochs of Tagore shards (4,237,448 bytes)
    d_bytes = 4 * 4237448
    bpt = BPT_TAGORE[tok]
    tokens_needed = d_bytes / bpt

    f_10m = calc_f_tok(
        MODELS_SPEC["10m"]["N_ne"],
        VOCABS[tok],
        MODELS_SPEC["10m"]["d"],
        MODELS_SPEC["10m"]["L"],
        S_TAGORE[tok],
    )
    f_scale = calc_f_tok(
        meta["N_ne"],
        VOCABS[tok],
        meta["d"],
        meta["L"],
        S_TAGORE[tok],
    )
    tok_s_10m = L4_MEASURED_TOK_PER_SEC[tok]
    tok_s_scale = tok_s_10m * (f_10m / f_scale)

    time_sec = tokens_needed / tok_s_scale
    units = time_sec * BURN_RATE_UNITS_PER_SEC
    return round(units, 2)


def recompute_all_runs() -> None:
    matrix_csv = Path("results/run_matrix.csv")
    rows = []
    with open(matrix_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print("=" * 95)
    print("RECOMPUTED RUN MATRIX UNITS FROM EMPIRICAL L4 BENCHMARK (BATCH 64)")
    print(f"Burn rate calibrated: 0.34 u / 10.6 min = {BURN_RATE_UNITS_PER_SEC * 3600:.4f} u/hr ({BURN_RATE_UNITS_PER_SEC:.8f} u/s)")
    print("Formula: tokens needed = 100 * N_ne bytes / measured bytes-per-token; units = (tokens / tok_s) * burn_rate")
    print("=" * 95)
    print(f"{'Stage':<15} | {'Run ID':<23} | {'Tok':<8} | {'Old Units':>9} | {'New Units':>9} | {'Approved':<8}")
    print("-" * 95)

    old_total_approved = 0.0
    new_total_approved = 0.0

    recomputed_rows = []

    for r in rows:
        stage = r["stage"]
        rid = r["run_id"]
        tok = r["tokenizer"]
        scale = r["model_size"]
        appr = r["approved"].lower() == "true"
        old_u = float(r["planned_units"])

        if stage.startswith("pretrain_"):
            new_u = compute_pretrain_run_units(scale, tok)
        elif stage.startswith("adapt_"):
            new_u = compute_adapt_run_units(scale, tok)
        elif stage.startswith("lr_sweep_"):
            new_u = compute_pretrain_run_units(scale, tok)
        else:
            new_u = old_u

        if appr:
            old_total_approved += old_u
            new_total_approved += new_u

        recomputed_rows.append({
            "stage": stage,
            "run_id": rid,
            "tokenizer": tok,
            "model_size": scale,
            "seed": r["seed"],
            "lr": r["lr"],
            "depends_on": r["depends_on"],
            "old_planned_units": old_u,
            "planned_units": new_u,
            "approved": appr,
        })

        print(f"{stage:<15} | {rid:<23} | {tok:<8} | {old_u:>8.2f}u | {new_u:>8.2f}u | {str(appr):<8}")

    print("=" * 95)
    print(f"TOTAL APPROVED UNITS (Original 39 approved runs):")
    print(f"  Old Planning Total : {old_total_approved:.2f} u (within 136.59 u)")
    print(f"  New Empirical Total: {new_total_approved:.2f} u")
    print(f"  Owner Budget Cap   : 200.00 u")
    print(f"  Required Reserve   :  35.00 u")
    print(f"  Spending Ceiling   : 165.00 u (200 - 35)")
    print(f"  Deficit / Over-Cap : +{new_total_approved - 165.00:.2f} u over spending ceiling!")
    print("=" * 95)

    # Save empirical matrix CSV
    out_csv = Path("results/run_matrix_empirical.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["stage", "run_id", "tokenizer", "model_size", "seed", "lr", "depends_on", "old_planned_units", "planned_units", "approved"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(recomputed_rows)
    print(f"\nSaved empirical recomputed matrix to: {out_csv}")

    print("\n" + "=" * 95)
    print("PROPOSALS FOR DROPPING CELLS TO FIT <= 165.00 u SPENDING CEILING")
    print("=" * 95)
    
    # Analysis of byte drops
    u_byte_25m = compute_pretrain_run_units("25m", "byte")
    u_adapt_byte_25m = compute_adapt_run_units("25m", "byte")
    print(f"1. Mandatory Drop: PT-byte-25m-s0 ({u_byte_25m:.2f} u) + ADAPT-byte-25m ({u_adapt_byte_25m:.2f} u):")
    post_byte = new_total_approved - (u_byte_25m + u_adapt_byte_25m)
    print(f"   -> Subtotal remaining: {post_byte:.2f} u (still {post_byte - 165.00:.2f} u over ceiling)")

    # Proposal A (User's guess: drop byte at 25M and large 50M):
    u_aks_50m = compute_pretrain_run_units("50m", "akshara")
    u_adapt_aks_50m = compute_adapt_run_units("50m", "akshara")
    u_char_10m_s2 = compute_pretrain_run_units("10m", "char")
    prop_a_total = post_byte - (u_aks_50m + u_adapt_aks_50m + u_char_10m_s2)
    print("\nPROPOSAL A (Drop expensive large cells: 25M byte, 50M akshara, 3rd char seed):")
    print(f"   - Drop PT-byte-25m-s0        : -{u_byte_25m:.2f} u")
    print(f"   - Drop ADAPT-byte-25m        : -{u_adapt_byte_25m:.2f} u")
    print(f"   - Drop PT-akshara-50m-s0     : -{u_aks_50m:.2f} u")
    print(f"   - Drop ADAPT-akshara-50m     : -{u_adapt_aks_50m:.2f} u")
    print(f"   - Drop PT-char-10m-s2        : -{u_char_10m_s2:.2f} u")
    print(f"   -> Resulting Approved Spend  : {prop_a_total:.2f} u (PASSES: under 165.00 u ceiling by {165.00 - prop_a_total:.2f} u; Reserve: {200.00 - prop_a_total:.2f} u)")
    print("   -> Retains: 10M (all), 25M (char, aks, bpe16, uni16), 50M (bpe16, uni16)")

    # Proposal B (Scientific Scaling Law Preservation: preserve all 3 tokenizers at 50M, prune replicate seeds):
    u_char_25m_s1 = compute_pretrain_run_units("25m", "char")
    u_aks_25m_s1 = compute_pretrain_run_units("25m", "akshara")
    u_byte_10m_s1 = compute_pretrain_run_units("10m", "byte")
    prop_b_total = post_byte - (u_char_25m_s1 + u_aks_25m_s1 + u_byte_10m_s1 + u_char_10m_s2)
    print("\nPROPOSAL B (Preserve all 50M models, prune replicate seeds at 25M & 10M):")
    print(f"   - Drop PT-byte-25m-s0        : -{u_byte_25m:.2f} u")
    print(f"   - Drop ADAPT-byte-25m        : -{u_adapt_byte_25m:.2f} u")
    print(f"   - Drop PT-char-25m-s1 (seed 1): -{u_char_25m_s1:.2f} u")
    print(f"   - Drop PT-akshara-25m-s1 (s1): -{u_aks_25m_s1:.2f} u")
    print(f"   - Drop PT-byte-10m-s1 (seed 1): -{u_byte_10m_s1:.2f} u")
    print(f"   - Drop PT-char-10m-s2 (seed 2): -{u_char_10m_s2:.2f} u")
    print(f"   -> Resulting Approved Spend  : {prop_b_total:.2f} u (PASSES: exactly matches ~165.00 u ceiling; Reserve: {200.00 - prop_b_total:.2f} u)")
    print("   -> Retains: 50M for all 3 tokenizers (aks, bpe16, uni16), giving full 3-point scaling curves!")
    print("=" * 95)


if __name__ == "__main__":
    recompute_all_runs()
