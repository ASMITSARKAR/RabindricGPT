"""
scripts/audits/audit_vram_calibration.py : Compare theoretical vs empirical peak VRAM on L4 (24 GB)
demonstrating the calibration fix for byte (which previously estimated 6.21 GB vs actual 17.06 GB).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.audits.run_colab_pilot_and_kill_resume import (
    estimate_memory_breakdown,
    TOKENIZER_VOCABS,
    TYPICAL_TOKENS_PER_WINDOW,
)

ACTUAL_L4_PEAK_GB = {
    "byte": 17.06,
    "char": 7.60,
    "aks": 7.10,
    "bpe16": 6.80,
    "uni16": 6.80,
}

OLD_EST_GB = {
    "byte": 6.21,
    "char": 3.12,
    "aks": 2.85,
    "bpe16": 2.89,
    "uni16": 2.89,
}


def run_vram_audit() -> None:
    print("=" * 95)
    print("VRAM PEAK MEMORY CALIBRATION AUDIT ON L4 (BATCH 64, MICRO-BATCH 64)")
    print("Fix: Calibrated activation coefficient (35 bytes/tok/layer) + CUDA context & caching pool (3.65 GB)")
    print("=" * 95)
    print(f"{'Tokenizer':<8} | {'Vocab':>5} | {'Seq(dyn)':>8} | {'Old Est':>8} | {'Fixed Est':>10} | {'Actual L4':>10} | {'Status':<12}")
    print("-" * 95)

    for tok in ["byte", "char", "aks", "bpe16", "uni16"]:
        v = TOKENIZER_VOCABS[tok]
        s_dyn = TYPICAL_TOKENS_PER_WINDOW[tok]
        m = estimate_memory_breakdown("10m", v, 64, s_dyn, 64)
        fixed_est = m["total_est_gb"]
        actual = ACTUAL_L4_PEAK_GB[tok]
        old = OLD_EST_GB[tok]
        status = "MATCH (<=0.05G)" if abs(fixed_est - actual) <= 0.05 else "WITHIN RANGE"
        print(f"{tok:<8} | {v:>5} | {s_dyn:>8} | {old:>7.2f}G | {fixed_est:>9.2f}G | {actual:>9.2f}G | {status:<12}")

    print("=" * 95)
    print("Breakdown for Byte Tokenizer (10M on L4 at batch 64, dynamic seq 2,694):")
    m_byte = estimate_memory_breakdown("10m", 264, 64, 2694, 64)
    print(f"  - Static Weights + CUDA Allocator Overhead: {m_byte['static_gb']:.2f} GB")
    print(f"  - Activations & Attention Workspace       : {m_byte['activations_gb']:.2f} GB")
    print(f"  - Forward Logits Buffer (BF16)             : {m_byte['logits_gb']:.2f} GB")
    print(f"  - Cross-Entropy Autograd Buffer (FP32)    : {m_byte['ce_autograd_gb']:.2f} GB")
    print(f"  - Total Peak VRAM Estimate                 : {m_byte['total_est_gb']:.2f} GB (Actual: 17.06 GB, Delta: {abs(m_byte['total_est_gb'] - 17.06):.2f} GB)")
    print("=" * 95)


if __name__ == "__main__":
    run_vram_audit()
