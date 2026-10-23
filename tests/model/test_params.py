"""
tests/model/test_params.py : Tests verifying exact agreement between analytic formulas and PyTorch numel counts.
"""

from pathlib import Path

from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import RabindricGPT
from src.rgpt.model.params import (
    compare_parameter_counts,
    count_parameters_empirical,
)


def test_analytic_vs_empirical_parameter_counts() -> None:
    """Analytic count must match sum(p.numel()) across all 4 model sizes and tokenizer vocabs."""
    for sz in ["10m", "25m", "50m", "100m"]:
        cfg_path = Path(f"configs/models/{sz}.yaml")
        assert cfg_path.exists()
        config = ModelConfig.from_yaml(cfg_path, vocab_size=8192)
        model = RabindricGPT(config)

        res = compare_parameter_counts(config, model)
        assert res["matches"] is True
        assert res["total_diff"] == 0
        assert res["non_embed_diff"] == 0


def test_non_embedding_parameters_excluded() -> None:
    """Non-embedding parameter count must strictly exclude tied token embedding weights."""
    config = ModelConfig(
        size_id="test",
        n_layers=2,
        d_model=128,
        head_dim=64,
        n_heads=2,
        n_kv_heads=2,
        hidden_dim=384,
        vocab_size=264,
        max_seq_len=512,
        tie_embeddings=True,
    )
    model = RabindricGPT(config)
    tot, ne = count_parameters_empirical(model)

    expected_embedding_params = config.vocab_size * config.d_model
    assert tot == ne + expected_embedding_params
