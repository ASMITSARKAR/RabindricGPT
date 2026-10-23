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

    # 1. 10M Main Ladder: 5 tokenizers x 3 seeds (seeds 0, 1, 2) = 15 runs
    tok_10m = [
        ("byte", 7.20),
        ("char", 1.60),
        ("akshara", 1.00),
        ("bpe16", 0.47),
        ("uni16", 0.47),
    ]
    for tok, units in tok_10m:
        for seed in [0, 1, 2]:
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

    # 2. 25M Main Ladder: 5 tokenizers x 1 seed (seed 0) + 1 BPE8 sensitivity = 6 runs
    tok_25m = [
        ("byte", 34.60),
        ("char", 8.30),
        ("akshara", 4.95),
        ("bpe16", 2.19),
        ("uni16", 2.19),
        ("bpe8", 2.24),
    ]
    for tok, units in tok_25m:
        matrix.append({
            "stage": "pretrain_25m",
            "run_id": f"PT-{tok}-25m-s0",
            "tokenizer": tok,
            "model_size": "25m",
            "seed": 0,
            "lr": 8e-4,
            "depends_on": f"PT-{tok}-10m-s0" if tok != "bpe8" else "PT-bpe16-10m-s0",
            "planned_units": units,
            "approved": True,
        })

    # 3. 50M Main Ladder: 4 tokenizers x 1 seed (seed 0) (byte excluded per DR-6) = 4 runs
    tok_50m = [
        ("char", 30.00),
        ("akshara", 17.50),
        ("bpe16", 7.50),
        ("uni16", 7.50),
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

    # 4. 100M Main Ladder: 2 tokenizers x 1 seed (seed 0) = 2 runs (GATED by 50M analysis)
    tok_100m = [
        ("bpe16", 28.30),
        ("uni16", 28.30),
    ]
    for tok, units in tok_100m:
        matrix.append({
            "stage": "pretrain_100m",
            "run_id": f"PT-{tok}-100m-s0",
            "tokenizer": tok,
            "model_size": "100m",
            "seed": 0,
            "lr": 4e-4,
            "depends_on": f"PT-{tok}-50m-s0",
            "planned_units": units,
            "approved": False,  # Gated
        })

    # 5. Equal-Compute Arms (byte, char, akshara matched to BPE compute at 10M and 25M) = 6 runs
    for size, units in [("10m", 0.47), ("25m", 2.19)]:
        for tok in ["byte", "char", "akshara"]:
            matrix.append({
                "stage": f"equal_compute_{size}",
                "run_id": f"EC-{tok}-{size}",
                "tokenizer": tok,
                "model_size": size,
                "seed": 0,
                "lr": 1e-3 if size == "10m" else 8e-4,
                "depends_on": f"PT-{tok}-{size}-s0",
                "planned_units": units,
                "approved": True,
            })

    # 6. Optional BPE-over-aksharas (G4 gated): 10M x 3 seeds, 25M, 50M = 5 runs
    bpeaks_runs = [
        ("10m", 0, 0.47),
        ("10m", 1, 0.47),
        ("10m", 2, 0.47),
        ("25m", 0, 2.19),
        ("50m", 0, 7.50),
    ]
    for size, seed, units in bpeaks_runs:
        matrix.append({
            "stage": f"bpeaks_{size}",
            "run_id": f"PT-bpeaks-{size}-s{seed}",
            "tokenizer": "bpeaks",
            "model_size": size,
            "seed": seed,
            "lr": 1e-3 if size == "10m" else (8e-4 if size == "25m" else 6e-4),
            "depends_on": "GATE_G4",
            "planned_units": units,
            "approved": False,  # Gated by G4
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
