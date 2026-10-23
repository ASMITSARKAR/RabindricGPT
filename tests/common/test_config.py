"""
tests/common/test_config.py : Tests for YAML configuration loader, unknown-key rejection, and hash stability.
"""

from pathlib import Path

import pytest

from src.rgpt.common.config import (
    ConfigError,
    compute_config_hash,
    load_config_with_decisions_override,
    load_decisions,
    load_yaml,
)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    """Unknown keys in YAML configuration must raise ConfigError."""
    bad_yaml = tmp_path / "bad_decisions.yaml"
    bad_yaml.write_text("unsupported_fake_key: 123\nchunk_size_chars: 1024\n")

    with pytest.raises(ConfigError, match="Unknown key"):
        load_decisions(bad_yaml)


def test_config_rejects_null_required_key(tmp_path: Path) -> None:
    """Null values for required keys must raise ConfigError."""
    null_yaml = tmp_path / "null_decisions.yaml"
    # Create valid decisions dict with one null field
    valid = load_yaml("configs/decisions.yaml")
    valid["init_std"] = None
    import yaml
    null_yaml.write_text(yaml.dump(valid))

    with pytest.raises(ConfigError, match="Null value"):
        load_decisions(null_yaml)


def test_load_production_decisions() -> None:
    """Production configs/decisions.yaml must load cleanly into frozen dataclass."""
    cfg = load_decisions()
    assert cfg.chunk_size_chars == 1024
    assert cfg.adapt_split == "Option_A"
    assert cfg.swiglu_multiple_of == 64
    assert cfg.fineweb2_target_bytes == 5.0e9


def test_config_override_order(tmp_path: Path) -> None:
    """Decisions overrides must take precedence over default base values."""
    base_file = tmp_path / "base.yaml"
    base_file.write_text("chunk_size_chars: 512\nlearning_rate: 0.001\n")

    override_file = tmp_path / "decisions_override.yaml"
    override_file.write_text("chunk_size_chars: 1024\n")

    merged = load_config_with_decisions_override(base_file, override_file)
    assert merged["chunk_size_chars"] == 1024
    assert merged["learning_rate"] == 0.001


def test_config_hash_stability() -> None:
    """Config hash must remain stable and deterministic for identical contents."""
    d1 = {"b": 2, "a": 1, "c": [1, 2, 3]}
    d2 = {"a": 1, "c": [1, 2, 3], "b": 2}
    assert compute_config_hash(d1) == compute_config_hash(d2)

    # Different content produces different hash
    d3 = {"a": 1, "b": 3}
    assert compute_config_hash(d1) != compute_config_hash(d3)
