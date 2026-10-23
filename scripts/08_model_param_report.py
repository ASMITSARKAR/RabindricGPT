"""
scripts/08_model_param_report.py : Generate parameter count report comparing analytic formula against PyTorch tensors.
Backlog/Phase : I7 / Phase H      Docs : 15_BUILD_DIRECTIVE.md Phase H; 08_COMPUTE_BUDGET.md §4
Contract      : configs/models/* -> results/params.md
Invariants    : Analytic count equals PyTorch sum(numel()) exactly for all sizes x tokenizers; reports delta vs 12*L*d^2
Failure modes : Discrepancy between analytic formula and empirical count
Status        : TESTED
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.common.atomic_io import atomic_write_text
from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import RabindricGPT
from src.rgpt.model.params import compare_parameter_counts, count_parameters_analytic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model parameter count report"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/params.md",
        help="Output Markdown report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_configs_dir = Path("configs/models")
    sizes = ["10m", "25m", "50m", "100m"]

    # Canonical tokenizer vocabs per 04_TOKENIZER_PLAN.md §1
    tokenizers = {
        "byte": 264,
        "char": 512,
        "aks": 8192,
        "bpe8": 8192,
        "bpe16": 16384,
        "uni16": 16384,
    }

    report_rows: List[Dict[str, Any]] = []
    all_matched = True

    print(
        f"{'Size':<6} | {'Tokenizer':<8} | {'V':<6} | {'Non-Embed N_ne':<15} | {'Embed V*d':<12} | {'Total N_tot':<15} | {'12*L*d^2':<12} | {'Match'}"
    )
    print("-" * 95)

    for sz in sizes:
        cfg_file = model_configs_dir / f"{sz}.yaml"
        if not cfg_file.exists():
            continue

        for tok_name, vocab_size in tokenizers.items():
            config = ModelConfig.from_yaml(cfg_file, vocab_size=vocab_size)
            model = RabindricGPT(config)

            cmp_res = compare_parameter_counts(config, model)
            if not cmp_res["matches"]:
                all_matched = False

            asymptotic_ne = 12 * config.n_layers * (config.d_model**2)
            tot, ne = count_parameters_analytic(config)
            embed_params = vocab_size * config.d_model

            row = {
                "size_id": sz,
                "n_layers": config.n_layers,
                "d_model": config.d_model,
                "hidden_dim": config.hidden_dim,
                "tokenizer": tok_name,
                "vocab_size": vocab_size,
                "non_embedding_params": ne,
                "embedding_params": embed_params,
                "total_params": tot,
                "asymptotic_12Ld2": asymptotic_ne,
                "delta_vs_asymptotic": ne - asymptotic_ne,
                "exact_match": cmp_res["matches"],
            }
            report_rows.append(row)

            match_str = "PASS" if cmp_res["matches"] else "FAIL"
            print(
                f"{sz:<6} | {tok_name:<8} | {vocab_size:<6} | {ne:<15,d} | {embed_params:<12,d} | {tot:<15,d} | {asymptotic_ne:<12,d} | {match_str}"
            )

    # Build Markdown report
    md_lines = [
        "# Model Parameter Accounting Report (Gate G3 / Directive Phase H)",
        "",
        "This table reports exact empirical PyTorch `sum(p.numel())` and analytic parameter counts across all model sizes and tokenizers.",
        "Non-embedding parameters $N_{ne}$ are completely independent of tokenizer vocabulary size $V$, validating fair compute allocation.",
        "",
        "| Size | Layers $L$ | Width $d$ | Hidden $d_{ff}$ | Tokenizer | Vocab $V$ | Non-Embedding $N_{ne}$ | Embedding $V \\cdot d$ | Total $N_{tot}$ | $12 L d^2$ | Delta | Match |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in report_rows:
        md_lines.append(
            f"| {r['size_id']} | {r['n_layers']} | {r['d_model']} | {r['hidden_dim']} | {r['tokenizer']} | {r['vocab_size']} | "
            f"{r['non_embedding_params']:,} | {r['embedding_params']:,} | {r['total_params']:,} | "
            f"{r['asymptotic_12Ld2']:,} | {r['delta_vs_asymptotic']:+,} | {'[x] PASS' if r['exact_match'] else '[ ] FAIL'} |"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, "\n".join(md_lines) + "\n")

    # Also save JSON report
    json_path = out_path.with_suffix(".json")
    atomic_write_text(
        json_path,
        json.dumps(
            {"all_matched": all_matched, "rows": report_rows},
            indent=2,
        ),
    )

    print(f"\nParameter report written to {out_path} and {json_path}")
    if not all_matched:
        raise ValueError(
            "Parameter mismatch detected between analytic formula and PyTorch module tensors!"
        )


if __name__ == "__main__":
    main()
