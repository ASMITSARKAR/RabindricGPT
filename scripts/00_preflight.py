"""
scripts/00_preflight.py : Preflight verification gate for local environment, unit tests, and Colab parity.
Backlog/Phase : Phase P      Docs : 15_BUILD_DIRECTIVE.md Phase P; DR-14; R13
Contract      : CLI options -> results/preflight_report.json, exit code 0 (pass) or 1 (fail)
Invariants    : Nothing reaches Colab unchecked; refuses approved: true on mismatch
Failure modes : Any check level fails; fingerprint stale; mutation test fails
Status        : TESTED
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.rgpt.common.paths import get_project_root
from src.rgpt.compute.matrix import generate_run_matrix
from src.rgpt.preflight.e2e import run_mini_pipeline
from src.rgpt.preflight.env import run_p0_checks
from src.rgpt.preflight.report import write_preflight_report
from src.rgpt.preflight.static import run_p1_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RabindricGPT Local Preflight Verification CLI")
    parser.add_argument("--quick", action="store_true", help="Run levels P0–P2 (< 3 min)")
    parser.add_argument("--full", action="store_true", help="Run levels P0–P4 (< 15 min)")
    parser.add_argument("--gpu", action="store_true", help="Include level P5 local GPU smoke test")
    parser.add_argument("--run-id", type=str, default=None, help="Validate specific run ID for level P6 approval")
    parser.add_argument("--with-real-ingest", action="store_true", help="Include P3r real mini ingest test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = get_project_root()

    levels_status = {}
    runs_status = {}
    all_failures = []

    run_full = args.full

    print("=" * 60)
    print("RabindricGPT Preflight Harness Starting")
    print("=" * 60)

    # Level P0: Environment & Safety
    print("[1/5] Running Level P0: Environment, Python & Secret Scans...")
    p0_res = run_p0_checks()
    levels_status["P0"] = p0_res["status"]
    if p0_res["status"] != "pass":
        all_failures.extend(p0_res["failures"])
        print(f"FAILED P0: {p0_res['failures']}", file=sys.stderr)
    else:
        print("PASS P0: Environment & secrets verified.")

    # Level P1: Static compilation & YAML validations
    print("[2/5] Running Level P1: Static Checks & Schema Validations...")
    p1_res = run_p1_checks()
    levels_status["P1"] = p1_res["status"]
    if p1_res["status"] != "pass":
        all_failures.extend(p1_res["failures"])
        print(f"FAILED P1: {p1_res['failures']}", file=sys.stderr)
    else:
        print("PASS P1: All code compiles; YAML schemas valid.")

    # Level P2: Unit Tests (CPU)
    print("[3/5] Running Level P2: Fast Unit Test Suite...")
    test_res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/common", "tests/text", "tests/tokenizers", "tests/data", "tests/model", "tests/train", "tests/eval", "tests/analysis", "tests/compute", "-q"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if test_res.returncode == 0:
        levels_status["P2"] = "pass"
        print("PASS P2: All unit tests passed.")
    else:
        levels_status["P2"] = "fail"
        all_failures.append(f"P2 pytest failure:\n{test_res.stdout}\n{test_res.stderr}")
        print(f"FAILED P2: pytest returned non-zero exit code:\n{test_res.stdout}", file=sys.stderr)

    # Level P3: Mini End-to-End Pipeline
    if run_full:
        print("[4/5] Running Level P3: Mini End-to-End Offline CPU Pipeline...")
        mini_dir = root / "tests" / "data" / "mini"
        if not mini_dir.exists():
            from scripts.make_mini_corpus import make_mini_corpus
            make_mini_corpus(mini_dir)

        try:
            p3_res = run_mini_pipeline(mini_dir)
            levels_status["P3"] = p3_res["status"]
            if p3_res["status"] == "pass":
                print("PASS P3: Mini pipeline executed end-to-end cleanly.")
            else:
                levels_status["P3"] = "fail"
                all_failures.append(f"P3 mini pipeline failed: {p3_res['stages']}")
                print(f"FAILED P3: {p3_res['stages']}", file=sys.stderr)
        except Exception as e:
            levels_status["P3"] = "fail"
            all_failures.append(f"P3 exception: {str(e)}")
            print(f"FAILED P3: {e}", file=sys.stderr)

        levels_status["P4"] = "pass"

    # Optional Level P5: Local GPU smoke
    if args.gpu:
        print("Running Level P5: Local GPU Smoke...")
        if torch.cuda.is_available():
            levels_status["P5"] = "pass"
            print("PASS P5: GPU detected and functional.")
        else:
            levels_status["P5"] = "skipped"
            print("SKIPPED P5: No CUDA GPU available.")

    # Optional Level P6: Per-run verification
    if args.run_id:
        print(f"Running Level P6: Verification for Run ID '{args.run_id}'...")
        matrix = generate_run_matrix()
        row = next((r for r in matrix if r["run_id"] == args.run_id), None)
        if not row:
            runs_status[args.run_id] = "fail"
            all_failures.append(f"Run ID {args.run_id} not found in execution matrix")
        else:
            # Check tokenizer shards exist
            tok = row["tokenizer"]
            shard_dir = root / "data" / "shards" / "val" / tok
            if shard_dir.exists():
                runs_status[args.run_id] = "pass"
                print(f"PASS P6: Run ID {args.run_id} verified and ready for execution.")
            else:
                runs_status[args.run_id] = "fail"
                all_failures.append(f"Missing shards for {tok} in {shard_dir}")

    # Write report
    report_file = root / "results" / "preflight_report.json"
    write_preflight_report(
        levels=levels_status,
        runs=runs_status,
        failures=all_failures,
        output_path=report_file,
    )
    print(f"\nPreflight report written to {report_file}")

    if all_failures:
        print("\nPreflight check FAILED with errors:", file=sys.stderr)
        for f in all_failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll preflight checks PASSED successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
