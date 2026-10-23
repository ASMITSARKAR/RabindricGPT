"""
scripts/14_update_ledger.py : Aggregate run metrics into compute ledger and update results/ledger.md.
Backlog/Phase : I14 / Phase L      Docs : 15_BUILD_DIRECTIVE.md Phase L; 08_COMPUTE_BUDGET.md §6
Contract      : runs/*/ledger_row.json -> results/ledger.md
Invariants    : Totals equal sum of individual run rows; never modifies docs without --patch-doc flag
Failure modes : Missing ledger rows; corrupted json
Status        : TESTED
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.common.paths import get_repo_root
from src.rgpt.compute.ledger import update_ledger_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update compute unit spend ledger")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing run logs")
    parser.add_argument("--output", type=str, default="results/ledger.md", help="Output ledger markdown path")
    parser.add_argument("--patch-doc", action="store_true", help="Patch doc 08 directly")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = repo_root / runs_dir

    out_file = Path(args.output)
    if not out_file.is_absolute():
        out_file = repo_root / out_file

    rep = update_ledger_report(runs_dir, out_file, patch_doc=args.patch_doc)
    print(f"Ledger updated: {rep['num_runs']} runs, total spent {rep['total_spent_units']} units -> {out_file}")


if __name__ == "__main__":
    main()
