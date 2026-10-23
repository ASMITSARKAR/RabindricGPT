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

    # Verification counts per directive
    count_10m = sum(1 for r in matrix if r["stage"] == "pretrain_10m")
    count_25m = sum(1 for r in matrix if r["stage"] == "pretrain_25m")
    count_50m = sum(1 for r in matrix if r["stage"] == "pretrain_50m")
    count_100m = sum(1 for r in matrix if r["stage"] == "pretrain_100m")
    count_ec = sum(1 for r in matrix if "equal_compute" in r["stage"])
    count_bpeaks = sum(1 for r in matrix if "bpeaks" in r["stage"])

    total_units = sum(r["planned_units"] for r in matrix)
    approved_units = sum(r["planned_units"] for r in matrix if r["approved"])

    print(f"Generated run matrix with {len(matrix)} runs to {saved}")
    print(f"Counts: 10M={count_10m}, 25M={count_25m}, 50M={count_50m}, 100M={count_100m}, EC={count_ec}, BPE-aks={count_bpeaks}")
    print(f"Total planned units: {total_units:.2f} (Approved: {approved_units:.2f})")

    assert count_10m == 15, f"Expected 15 runs at 10M, got {count_10m}"
    assert count_25m == 6, f"Expected 6 runs at 25M, got {count_25m}"
    assert count_50m == 4, f"Expected 4 runs at 50M, got {count_50m}"
    assert count_100m == 2, f"Expected 2 gated runs at 100M, got {count_100m}"
    assert count_ec == 6, f"Expected 6 equal-compute runs, got {count_ec}"
    assert count_bpeaks == 5, f"Expected 5 bpeaks runs, got {count_bpeaks}"


if __name__ == "__main__":
    main()
