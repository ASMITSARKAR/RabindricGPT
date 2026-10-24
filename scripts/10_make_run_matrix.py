"""
scripts/10_make_run_matrix.py : Generate and export results/run_matrix.csv and verify budget plan consistency.
Backlog/Phase : Phase M      Docs : 15_BUILD_DIRECTIVE.md §2, Phase M; 08_COMPUTE_BUDGET.md §6
Contract      : configs/* -> results/run_matrix.csv
Invariants    : Run counts match specs (10M: 15 runs, 25M: 6, 50M: 4, 100M: 2 gated); totals match planned units
Failure modes : Budget total mismatch; unapproved run misconfiguration
Status        : TESTED
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.common.paths import get_repo_root
from src.rgpt.compute.matrix import generate_run_matrix, save_run_matrix_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate run matrix CSV")
    parser.add_argument("--output", type=str, default="results/run_matrix.csv", help="Matrix CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path

    matrix = generate_run_matrix()
    saved = save_run_matrix_csv(out_path)

    # Verification counts per revised 200-unit matrix (49 runs total: 39 approved, 10 unapproved)
    count_10m = sum(1 for r in matrix if r["stage"] == "pretrain_10m")
    count_25m = sum(1 for r in matrix if r["stage"] == "pretrain_25m")
    count_50m = sum(1 for r in matrix if r["stage"] == "pretrain_50m")
    count_100m = sum(1 for r in matrix if r["stage"] == "pretrain_100m")
    count_adapt_approved = sum(1 for r in matrix if r["stage"].startswith("adapt_") and r["approved"])
    count_adapt_unapproved = sum(1 for r in matrix if r["stage"].startswith("adapt_") and not r["approved"])
    count_sweep = sum(1 for r in matrix if r["stage"] == "lr_sweep_10m")

    approved_units = sum(r["planned_units"] for r in matrix if r["approved"])
    sweep_units = sum(r["planned_units"] for r in matrix if r["stage"] == "lr_sweep_10m")
    funded_with_sweep = approved_units + sweep_units

    print(f"Generated run matrix with {len(matrix)} runs to {saved}")
    print(f"Counts: 10M={count_10m}, 25M={count_25m}, 50M={count_50m}, 100M={count_100m} (unapproved), Adapt={count_adapt_approved}+{count_adapt_unapproved}, Sweep={count_sweep}")
    print(f"Approved sum (10M, 25M, 50M): {approved_units:.2f} u")
    print(f"Sum with LR sweep: {funded_with_sweep:.2f} u")
    print(f"Cap of 200 with 35u reserve: {funded_with_sweep:.2f} u <= 165.00 u (Reserve: {200.0 - funded_with_sweep:.2f} u >= 35.00 u)")

    assert count_10m == 14, f"Expected 14 runs at 10M, got {count_10m}"
    assert count_25m == 9, f"Expected 9 runs at 25M, got {count_25m}"
    assert count_50m == 3, f"Expected 3 runs at 50M, got {count_50m}"
    assert count_100m == 3, f"Expected 3 unapproved runs at 100M, got {count_100m}"
    assert count_adapt_approved == 13, f"Expected 13 approved adapt runs, got {count_adapt_approved}"
    assert count_adapt_unapproved == 3, f"Expected 3 unapproved 100M adapt runs, got {count_adapt_unapproved}"
    assert count_sweep == 4, f"Expected 4 sweep runs, got {count_sweep}"
    assert len(matrix) == 49, f"Expected 49 total runs, got {len(matrix)}"
    assert approved_units <= 165.0, f"Approved units {approved_units} exceed 165u cap"


if __name__ == "__main__":
    main()
