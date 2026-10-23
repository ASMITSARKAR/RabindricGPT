"""
tests/model/test_config.py : Tests for ModelConfig validation, dimension rules, and YAML loading.
"""

from pathlib import Path

import pytest

from src.rgpt.model.config import ModelConfig


def test_model_config_head_dimension_rule() -> None:
    """n_heads must equal d_model / 64 for all configurations."""
    cfg = ModelConfig(
        size_id="test",
        n_layers=4,
        d_model=256,
        head_dim=64,
        n_heads=4,
        n_kv_heads=4,
        hidden_dim=704,  # multiple of 64
        vocab_size=1000,
        max_seq_len=512,
    )
    assert cfg.n_heads == cfg.d_model // cfg.head_dim

    # Non-matching n_heads must raise ValueError
    with pytest.raises(ValueError, match="must equal d_model // head_dim"):
        ModelConfig(
            size_id="test",
            n_layers=4,
            d_model=256,
            head_dim=64,
            n_heads=8,  # Mismatch: 256/64 is 4
            n_kv_heads=4,
            hidden_dim=704,
            vocab_size=1000,
            max_seq_len=512,
        )


def test_model_config_swiglu_hidden_rounding() -> None:
    """hidden_dim must be rounded up to a multiple of 64 per DR-3."""
    # Invalid hidden_dim not divisible by 64
    with pytest.raises(ValueError, match="multiple of 64"):
        ModelConfig(
            size_id="test",
            n_layers=4,
            d_model=256,
            head_dim=64,
            n_heads=4,
            n_kv_heads=4,
            hidden_dim=700,  # Not multiple of 64
            vocab_size=1000,
            max_seq_len=512,
        )

    # Valid config from yaml
    yaml_file = Path("configs/models/10m.yaml")
    assert yaml_file.exists()
    cfg = ModelConfig.from_yaml(yaml_file, vocab_size=264)
    assert cfg.hidden_dim % 64 == 0
