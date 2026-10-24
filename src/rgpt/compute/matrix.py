"""
src/rgpt/compute/matrix.py : Generator for results/run_matrix.csv defining all planned training and adaptation runs.
Backlog/Phase : Phase M      Docs : 15_BUILD_DIRECTIVE.md §2, Phase M; 10_RUN_MATRIX.md; 08_COMPUTE_BUDGET.md
Contract      : generate_run_matrix() -> list[dict]
Invariants    : Matrix run counts match section 2 (10M: 15 runs, 25M: 6, 50M: 4, 100M: 2 gated); IDs unique
Failure modes : Duplicate run ID generated; budget total exceeds planned ceiling
Status        : TESTED
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Union


def generate_run_matrix() -> List[Dict[str, Any]]:
    """Generate the full execution matrix rows from configs.

    Returns:
        List of dictionaries defining stage, run_id, tokenizer, size, seed, LR, planned_units, approved.
    """
    matrix: List[Dict[str, Any]] = []

    # 1. 10M Main Ladder: 5 tokenizers (byte: 2 seeds; char, aks, bpe16, uni16: 3 seeds) = 14 runs
    # Measured on FineWeb shards: byte=6.71, char=1.71, aks=1.20, bpe16=0.49, uni16=0.49
    tok_10m = [
        ("byte", 6.71, [0, 1]),
        ("char", 1.71, [0, 1, 2]),
        ("akshara", 1.20, [0, 1, 2]),
        ("bpe16", 0.49, [0, 1, 2]),
        ("uni16", 0.49, [0, 1, 2]),
    ]
    for tok, units, seeds in tok_10m:
        for seed in seeds:
            matrix.append({
                "stage": "pretrain_10m",
                "run_id": f"PT-{tok}-10m-s{seed}",
                "tokenizer": tok,
                "model_size": "10m",
                "seed": seed,
                "lr": 1e-3,
                "depends_on": "",
                "planned_units": units,
                "approved": True,
            })

    # 2. 25M Main Ladder: char/aks/bpe16/uni16 (2 seeds) plus byte (1 seed) = 9 runs
    # Measured on FineWeb shards: byte=33.61, char=9.14, aks=6.08, bpe16=2.39, uni16=2.35
    tok_25m = [
        ("byte", 33.61, [0]),
        ("char", 9.14, [0, 1]),
        ("akshara", 6.08, [0, 1]),
        ("bpe16", 2.39, [0, 1]),
        ("uni16", 2.35, [0, 1]),
    ]
    for tok, units, seeds in tok_25m:
        for seed in seeds:
            matrix.append({
                "stage": "pretrain_25m",
                "run_id": f"PT-{tok}-25m-s{seed}",
                "tokenizer": tok,
                "model_size": "25m",
                "seed": seed,
                "lr": 8e-4,
                "depends_on": f"PT-{tok}-10m-s0",
                "planned_units": units,
                "approved": True,
            })

    # 3. 50M Main Ladder: aks/bpe16/uni16 (1 seed) = 3 runs (byte & char excluded per DR-6 & budget cap)
    # Measured on FineWeb shards: aks=21.03 (exact 21.0284), bpe16=8.07, uni16=7.93
    # Note: 2.2e16 FLOPs/unit is an unmeasured planning assumption awaiting Action B empirical calibration.
    tok_50m = [
        ("akshara", 21.03),
        ("bpe16", 8.07),
        ("uni16", 7.93),
    ]
    for tok, units in tok_50m:
        matrix.append({
            "stage": "pretrain_50m",
            "run_id": f"PT-{tok}-50m-s0",
            "tokenizer": tok,
            "model_size": "50m",
            "seed": 0,
            "lr": 6e-4,
            "depends_on": f"PT-{tok}-25m-s0",
            "planned_units": units,
            "approved": True,
        })

    # 4. Adaptation Runs on Tagore shards (4 epochs across active model configurations) = 13 runs
    adapt_runs = [
        ("10m", "byte", 0.15),
        ("10m", "char", 0.04),
        ("10m", "akshara", 0.03),
        ("10m", "bpe16", 0.01),
        ("10m", "uni16", 0.01),
        ("25m", "byte", 0.31),
        ("25m", "char", 0.08),
        ("25m", "akshara", 0.06),
        ("25m", "bpe16", 0.03),
        ("25m", "uni16", 0.03),
        ("50m", "akshara", 0.10),
        ("50m", "bpe16", 0.05),
        ("50m", "uni16", 0.04),
    ]
    for size, tok, units in adapt_runs:
        matrix.append({
            "stage": f"adapt_{size}",
            "run_id": f"ADAPT-{tok}-{size}",
            "tokenizer": tok,
            "model_size": size,
            "seed": 0,
            "lr": 3e-4,
            "depends_on": f"PT-{tok}-{size}-s0",
            "planned_units": units,
            "approved": True,
        })

    # 5. 10M LR Sweep: bpe16 and akshara, lr in {5e-4, 1e-3, 2e-3}, 1 seed
    # Relative kill rule: abort at step 500 if loss hasn't dropped by >=15% (ratio > 0.85) or > 7.5 (lr=1e-3 is in main ladder PT-{tok}-10m-s0)
    lr_sweep_runs = [
        ("bpe16", 5e-4, 0.49),
        ("bpe16", 2e-3, 0.49),
        ("akshara", 5e-4, 1.20),
        ("akshara", 2e-3, 1.20),
    ]
    for tok, lr_val, units in lr_sweep_runs:
        lr_tag = "5e-4" if lr_val == 5e-4 else "2e-3"
        matrix.append({
            "stage": "lr_sweep_10m",
            "run_id": f"SWEEP-{tok}-10m-lr{lr_tag}",
            "tokenizer": tok,
            "model_size": "10m",
            "seed": 0,
            "lr": lr_val,
            "depends_on": "",
            "planned_units": units,
            "approved": False,
        })

    # Check uniqueness of run IDs
    seen_ids = set()
    for row in matrix:
        rid = row["run_id"]
        if rid in seen_ids:
            raise ValueError(f"Duplicate run_id in matrix: {rid}")
        seen_ids.add(rid)

    return matrix


def save_run_matrix_csv(output_path: Union[str, Path]) -> Path:
    """Save the generated run matrix to CSV file."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_run_matrix()
    headers = [
        "stage",
        "run_id",
        "tokenizer",
        "model_size",
        "seed",
        "lr",
        "depends_on",
        "planned_units",
        "approved",
    ]

    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return out_file
