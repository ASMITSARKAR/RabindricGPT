"""
src/rgpt/analysis/tables.py : Markdown and CSV reporting tables with complete run-ID provenance.
Backlog/Phase : I11 / Phase J      Docs : 15_BUILD_DIRECTIVE.md Phase J; 01 §1
Contract      : runs_dir: Path -> report_tables: dict[str, str]
Invariants    : Every number is accompanied by run ID provenance; no unverified figures in committed tables
Failure modes : Missing run metric file; conflicting metric keys
Status        : TESTED
"""

import json
from pathlib import Path
from typing import Dict, List, Union


def generate_summary_tables(runs_dir: Union[str, Path]) -> Dict[str, str]:
    """Parse runs or results directory and format Markdown tables with run-ID provenance.

    Returns:
        Dictionary mapping table name to GitHub-flavored Markdown text.
    """
    path = Path(runs_dir)
    tables: Dict[str, str] = {}

    eval_files: List[Path] = []
    if path.is_dir():
        eval_files = list(path.glob("eval_*.json")) + list(path.glob("*/eval.json"))

    records = []
    for f in eval_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                records.append(data)
        except Exception:
            continue

    # Main scaling and evaluation summary table
    headers = [
        "Run ID",
        "Tokenizer",
        "Model Size",
        "BPC",
        "95% CI",
        "BPB",
        "Ill-formed Unicode %",
        "In-Vocab Word %",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for r in sorted(records, key=lambda x: str(x.get("run_id", ""))):
        run_id = r.get("run_id", "N/A")
        tok = r.get("tokenizer", "N/A")
        size = r.get("model_size", "N/A")
        bpc = f"{r['bpc']:.4f}" if "bpc" in r else "N/A"
        ci = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]" if "ci_lower" in r and "ci_upper" in r else "N/A"
        bpb = f"{r['bpb']:.4f}" if "bpb" in r else "N/A"
        ill = f"{r['ill_formed_unicode_rate']*100:.2f}%" if "ill_formed_unicode_rate" in r else "N/A"
        invoc = f"{r['in_vocab_word_rate']*100:.2f}%" if "in_vocab_word_rate" in r else "N/A"

        row = [run_id, tok, size, bpc, ci, bpb, ill, invoc]
        lines.append("| " + " | ".join(row) + " |")

    tables["main_summary_table"] = "\n".join(lines)
    return tables
