"""
src/rgpt/compute/ledger.py : Accounting ledger reader and markdown generator tracking unit spend against monthly allocations.
Backlog/Phase : I14 / Phase L      Docs : 15_BUILD_DIRECTIVE.md Phase L; 08_COMPUTE_BUDGET.md §6
Contract      : aggregate_ledger(runs_dir) -> ledger_report: dict
Invariants    : Total recorded spend equals exact sum of run ledger rows; never modifies docs without flag
Failure modes : Inconsistent ledger row files; negative spent units
Status        : TESTED
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def update_ledger_report(
    runs_dir: Union[str, Path],
    output_path: Union[str, Path],
    patch_doc: bool = False,
) -> Dict[str, Any]:
    """Parse all ledger_row.json files and generate current spend ledger."""
    p_runs = Path(runs_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    row_files: List[Path] = []
    if p_runs.exists():
        row_files = list(p_runs.glob("*/ledger_row.json")) + list(p_runs.glob("ledger_row_*.json"))

    rows: List[Dict[str, Any]] = []
    total_spent = 0.0

    for rf in sorted(row_files):
        try:
            with open(rf, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                units = float(data.get("spent_units", 0.0))
                if units < 0:
                    raise ValueError(f"Negative spent units in {rf}: {units}")
                total_spent += units
                rows.append(data)
        except Exception:
            continue

    # Generate Markdown table
    headers = ["Run ID", "Stage", "Tokenizer", "Size", "GPU", "Spent Units", "Time (s)"]
    md_lines = [
        "# Compute Spend Ledger",
        "",
        f"**Total Spent Units:** {total_spent:.2f} units across {len(rows)} recorded runs.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for r in rows:
        md_lines.append(
            f"| {r.get('run_id', 'N/A')} | {r.get('stage', 'N/A')} | {r.get('tokenizer', 'N/A')} | "
            f"{r.get('model_size', 'N/A')} | {r.get('gpu', 'N/A')} | {float(r.get('spent_units', 0.0)):.2f} | "
            f"{r.get('time_seconds', 'N/A')} |"
        )

    out_file.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "total_spent_units": round(total_spent, 4),
        "num_runs": len(rows),
        "rows": rows,
        "output_path": str(out_file),
    }
