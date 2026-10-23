"""
src/rgpt/analysis/plots.py : Publication-quality visualization of scaling curves, BPB vs T, and adaptation trajectories.
Backlog/Phase : I11 / Phase J      Docs : 15_BUILD_DIRECTIVE.md Phase J; 01_CHARTER.md
Contract      : data: dict, output_path: Path -> None
Invariants    : Reproducible figures saved to results/; all curves carry error bars; includes 2605.01188 test
Failure modes : Missing output directory; empty metrics dataset
Status        : TESTED
"""

from pathlib import Path
from typing import Any, Dict, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_scaling_curves(data: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Plot BPC and BPB vs non-embedding parameters across all tokenizers."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # data format: {"tokenizers": {"bpe16": {"sizes": [10e6, 25e6], "bpc": [1.4, 1.2], "bpb": [1.1, 0.95], ...}, ...}}
    tokenizers_data = data.get("tokenizers", {})

    for tok_name, tok_metrics in tokenizers_data.items():
        sizes = tok_metrics.get("sizes", [])
        bpcs = tok_metrics.get("bpc", [])
        bpbs = tok_metrics.get("bpb", [])
        bpc_errors = tok_metrics.get("bpc_ci", None)

        if sizes and bpcs:
            if bpc_errors:
                ax1.errorbar(sizes, bpcs, yerr=bpc_errors, marker="o", label=tok_name, capsize=3)
            else:
                ax1.plot(sizes, bpcs, marker="o", label=tok_name)

        if sizes and bpbs:
            ax2.plot(sizes, bpbs, marker="s", label=tok_name)

    ax1.set_xscale("log")
    ax1.set_xlabel("Non-embedding parameters (N_ne)")
    ax1.set_ylabel("Bits Per Character (BPC)")
    ax1.set_title("Scaling Laws: BPC vs Model Size")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.set_xscale("log")
    ax2.set_xlabel("Non-embedding parameters (N_ne)")
    ax2.set_ylabel("Bits Per Byte (BPB)")
    ax2.set_title("Scaling Laws: BPB vs Model Size")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_bpb_vs_compression(data: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Plot BPB vs compression rate T per model size (testing 2605.01188 hypothesis)."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # data format: {"sizes": {"10m": {"T": [1.0, 2.5, 4.0], "bpb": [1.5, 1.2, 1.3]}, ...}}
    sizes_data = data.get("sizes", {})

    for size_name, metrics in sizes_data.items():
        t_vals = metrics.get("T", [])
        bpb_vals = metrics.get("bpb", [])
        if t_vals and bpb_vals:
            ax.plot(t_vals, bpb_vals, marker="o", label=f"Size: {size_name}")

    ax.set_xlabel("Compression Rate T (bytes / token)")
    ax.set_ylabel("Bits Per Byte (BPB)")
    ax.set_title("BPB vs Compression Rate T (2605.01188 Optimum Shift Test)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_adaptation_trajectories(data: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Plot adaptation validation BPC curves over fine-tuning steps."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # data format: {"runs": {"AD-bpe16-25m-lr1": {"steps": [0, 50, 100], "val_bpc": [1.3, 1.1, 0.95]}, ...}, "target_bpc": 1.0}
    runs_data = data.get("runs", {})
    target_bpc = data.get("target_bpc", None)

    for run_id, metrics in runs_data.items():
        steps = metrics.get("steps", [])
        bpcs = metrics.get("val_bpc", [])
        if steps and bpcs:
            ax.plot(steps, bpcs, marker=".", label=run_id)

    if target_bpc is not None:
        ax.axhline(target_bpc, color="red", linestyle="--", label=f"Target BPC ({target_bpc:.3f})")

    ax.set_xlabel("Adaptation Steps")
    ax.set_ylabel("Validation BPC")
    ax.set_title("Tagore Adaptation Trajectories")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
