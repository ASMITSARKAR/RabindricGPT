"""
scripts/04_audit.py : Generate Gate G1 data audit report and verify corpus acceptance criteria.
Backlog/Phase : I5 / Phase F      Docs : 15_BUILD_DIRECTIVE.md Phase F; 03_DATA_PLAN.md §9
Contract      : data/cleaned/* -> results/g1_audit.json, results/g1_audit.md
Invariants    : All rows of 03 §9 verified; bytes/char and chars/akshara reported
Failure modes : Quality 0-2 page detected; unstripped wikitext templates found; ratio < 80%
Status        : TESTED
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.data.audit import generate_audit_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data audit report for Gate G1")
    parser.add_argument(
        "--wikisource-dir",
        type=str,
        default="data/cleaned/tagore",
        help="Path to cleaned Wikisource Tagore texts",
    )
    parser.add_argument(
        "--fineweb2-dir",
        type=str,
        default="data/cleaned/fineweb2",
        help="Path to cleaned FineWeb-2 texts",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Report output directory"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Executing Gate G1 Data Audit...")
    report = generate_audit_report(
        wikisource_dir=args.wikisource_dir,
        fineweb2_dir=args.fineweb2_dir,
        output_dir=args.output_dir,
    )

    print(f"Gate G1 Status: {'PASSED' if report['passed'] else 'FAILED'}")
    print(f"Total Characters: {report['total_chars']:,}")
    print(f"Total Bytes: {report['total_bytes']:,}")
    print(f"Bytes/Char: {report['bytes_per_char']:.4f}")
    print(f"Chars/Akshara: {report['chars_per_akshara']:.4f}")
    print(f"Bengali Ratio: {report['bengali_script_ratio'] * 100:.2f}%")
    print(f"Audit report saved to: {args.output_dir}/g1_audit.json and g1_audit.md")


if __name__ == "__main__":
    main()
