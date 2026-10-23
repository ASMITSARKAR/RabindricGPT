"""
tests/preflight/test_memory.py : Tests for analytic GPU memory estimation and safety margin checks.
"""

from src.rgpt.model.config import ModelConfig
from src.rgpt.preflight.memory import estimate_vram_usage_gb


def test_memory_estimator_bounds() -> None:
    """Estimated VRAM must account for params, AdamW states, gradients, and activation buffers."""
    cfg = ModelConfig(
        size_id="10m",
        n_layers=6,
        d_model=384,
        head_dim=64,
        n_heads=6,
        n_kv_heads=6,
        hidden_dim=1024,
        vocab_size=16384,
        max_seq_len=512,
    )

    mem = estimate_vram_usage_gb(cfg, batch_size=2, s_max=512, precision="bf16", optimizer="adamw")

    assert mem["params_gb"] > 0
    assert mem["optimizer_gb"] > mem["params_gb"]  # AdamW is 12 bytes/param vs 2 bytes/param
    assert mem["gradients_gb"] > 0
    assert mem["activations_gb"] > 0

    expected_sum = (
        mem["params_gb"]
        + mem["optimizer_gb"]
        + mem["gradients_gb"]
        + mem["activations_gb"]
    )
    assert abs(mem["total_gb"] - expected_sum) < 1e-3

    # 10M with batch size 2 must easily fit in 16GB and 24GB
    assert mem["fits_l4_80pct"]
    assert mem["fits_t4_80pct"]
