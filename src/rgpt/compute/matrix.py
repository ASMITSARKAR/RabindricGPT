"""
src/rgpt/compute/matrix.py : Generator for results/run_matrix.csv defining all planned training and adaptation runs.
Backlog/Phase : Phase M      Docs : 15_BUILD_DIRECTIVE.md §2, Phase M; 10_RUN_MATRIX.md; 08_COMPUTE_BUDGET.md
Contract      : generate_run_matrix() -> list[dict]
Invariants    : Approved runs: 10M all 5 (byte 2 seeds, others 3 seeds = 14), 25M all 5 (1 seed each = 5),
                50M akshara and bpe16 only (1 seed each = 2); 12 approved adapt runs; total <= 150 u.
Failure modes : Duplicate run ID generated; budget total exceeds planned ceiling
Status        : TESTED (Empirical Calibration Approved Set)
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

    # 1. 10M Main Ladder: 5 tokenizers (byte: 2 seeds; char, aks, bpe16, uni16: 3 seeds) = 14 approved runs
    # Calibrated from measured tokens/s on L4 (burn rate 0.34u / 10.6 min):
    # byte=11.99, char=3.70, aks=2.63, bpe16=1.06, uni16=1.04
    tok_10m = [
        ("byte", 11.99, [0, 1]),
        ("char", 3.70, [0, 1, 2]),
        ("akshara", 2.63, [0, 1, 2]),
        ("bpe16", 1.06, [0, 1, 2]),
        ("uni16", 1.04, [0, 1, 2]),
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

    # 2. 25M Main Ladder: all 5 tokenizers, 1 seed each approved (seed 0); seed 1 unapproved
    # Calibrated from measured throughput: byte=29.00, char=8.94, aks=6.36, bpe16=2.56, uni16=2.51
    tok_25m = [
        ("byte", 29.00, [0]),
        ("char", 8.94, [0, 1]),
        ("akshara", 6.36, [0, 1]),
        ("bpe16", 2.56, [0, 1]),
        ("uni16", 2.51, [0, 1]),
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
                "approved": (seed == 0),  # 1 seed approved per user directive
            })

    # 3. 50M Main Ladder: akshara and bpe16 only, 1 seed each (seed 0); uni16 unapproved
    # Calibrated from measured throughput: aks=12.26, bpe16=4.93, uni16=4.85
    tok_50m = [
        ("akshara", 12.26, True),
        ("bpe16", 4.93, True),
        ("uni16", 4.85, False),
    ]
    for tok, units, is_appr in tok_50m:
        matrix.append({
            "stage": "pretrain_50m",
            "run_id": f"PT-{tok}-50m-s0",
            "tokenizer": tok,
            "model_size": "50m",
            "seed": 0,
            "lr": 6e-4,
            "depends_on": f"PT-{tok}-25m-s0",
            "planned_units": units,
            "approved": is_appr,
        })

    # 4. Adaptation Runs on Tagore shards (4 epochs) = 12 approved runs
    adapt_runs = [
        ("10m", "byte", 0.19, True),
        ("10m", "char", 0.06, True),
        ("10m", "akshara", 0.04, True),
        ("10m", "bpe16", 0.02, True),
        ("10m", "uni16", 0.02, True),
        ("25m", "byte", 0.40, True),
        ("25m", "char", 0.13, True),
        ("25m", "akshara", 0.09, True),
        ("25m", "bpe16", 0.04, True),
        ("25m", "uni16", 0.04, True),
        ("50m", "akshara", 0.16, True),
        ("50m", "bpe16", 0.07, True),
        ("50m", "uni16", 0.07, False),  # unapproved matching 50M pretrain
    ]
    for size, tok, units, is_appr in adapt_runs:
        matrix.append({
            "stage": f"adapt_{size}",
            "run_id": f"ADAPT-{tok}-{size}",
            "tokenizer": tok,
            "model_size": size,
            "seed": 0,
            "lr": 3e-4,
            "depends_on": f"PT-{tok}-{size}-s0",
            "planned_units": units,
            "approved": is_appr,
        })

    # 5. 10M LR Sweep: unapproved
    lr_sweep_runs = [
        ("bpe16", 5e-4, 1.06),
        ("bpe16", 2e-3, 1.06),
        ("akshara", 5e-4, 2.63),
        ("akshara", 2e-3, 2.63),
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

    # 6. 100M Stage: unapproved
    tok_100m = [
        ("akshara", 24.52),
        ("bpe16", 9.86),
        ("uni16", 9.70),
    ]
    for tok, units in tok_100m:
        matrix.append({
            "stage": "pretrain_100m",
            "run_id": f"PT-{tok}-100m-s0",
            "tokenizer": tok,
            "model_size": "100m",
            "seed": 0,
            "lr": 5e-4,
            "depends_on": f"PT-{tok}-50m-s0",
            "planned_units": units,
            "approved": False,
        })

    adapt_100m = [
        ("100m", "akshara", 0.32),
        ("100m", "bpe16", 0.14),
        ("100m", "uni16", 0.14),
    ]
    for size, tok, units in adapt_100m:
        matrix.append({
            "stage": f"adapt_{size}",
            "run_id": f"ADAPT-{tok}-{size}",
            "tokenizer": tok,
            "model_size": size,
            "seed": 0,
            "lr": 3e-4,
            "depends_on": f"PT-{tok}-{size}-s0",
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
