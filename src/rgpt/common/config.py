"""
src/rgpt/common/config.py : YAML configuration loader into frozen dataclasses with unknown-key rejection and hash stability.
Backlog/Phase : I-none / Phase B      Docs : 15_BUILD_DIRECTIVE.md Phase B
Contract      : yaml_path: Path -> FrozenConfig dataclass
Invariants    : Unknown keys raise ConfigError; null required-at-gate keys raise; stable sha256 config hash
Failure modes : Invalid YAML syntax; missing file; unknown schema keys; type mismatches
Status        : TESTED
"""

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml

from src.rgpt.common.paths import resolve_path

T = TypeVar("T")


class ConfigError(Exception):
    """Raised when configuration validation or loading fails."""
    pass


@dataclass(frozen=True)
class DecisionsConfig:
    """Frozen dataclass representation of decisions.yaml."""
    chunk_size_chars: int
    chunk_fallback_chars: int
    eval_prefix_bos: bool
    eval_append_eos: bool
    score_bos: bool
    swiglu_multiple_of: int
    init_std: float
    embedding_init_scale: float
    scale_residual_by_depth: bool
    tie_embeddings: bool
    use_bias: bool
    batch_size_chunks: int
    adamw_beta1: float
    adamw_beta2: float
    adamw_weight_decay: float
    grad_clip: float
    warmup_ratio: float
    min_lr_ratio: float
    lr_sweep_10m: list
    pad_loss_masked: bool
    gen_temperature: float
    gen_top_p: float
    gen_max_new_chars: int
    gen_fixed_seeds: list
    target_bpc_rule: str
    pua_start: int
    pua_end: int
    adapt_epochs: int
    adapt_split: str
    shard_volume_margin: float
    runner_max_attempts: int
    require_preflight_pass: bool
    fineweb2_target_bytes: float
    dedup_max_ram_gb: float
    colab_reserve_units: int
    default_lr: float = 0.0010
    default_lr_10m: float = 0.0010
    dr_10_amendment: Optional[Dict[str, Any]] = None


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load raw YAML content from disk.

    Args:
        path: Path to YAML file.

    Returns:
        Dictionary of configuration values.

    Raises:
        ConfigError: If file not found or invalid YAML syntax.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise ConfigError(f"Configuration file not found: {resolved}")
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
            return content if content is not None else {}
    except Exception as e:
        raise ConfigError(f"Failed to parse YAML file {resolved}: {e}") from e


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of configuration dictionary.

    Args:
        config_dict: Dictionary to hash.

    Returns:
        Hex-encoded SHA256 string.
    """
    normalized_json = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()


def bind_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Instantiate a dataclass from a dict, raising ConfigError on unknown or missing keys."""
    valid_fields = {f.name for f in fields(cls)}
    provided_keys = set(data.keys())

    unknown_keys = provided_keys - valid_fields
    if unknown_keys:
        raise ConfigError(
            f"Unknown key(s) in configuration for {cls.__name__}: {sorted(list(unknown_keys))}"
        )

    # Check for missing required fields
    init_kwargs = {}
    for f in fields(cls):
        if f.name in data:
            val = data[f.name]
            if val is None:
                raise ConfigError(
                    f"Null value for required configuration field '{f.name}' in {cls.__name__}"
                )
            init_kwargs[f.name] = val
        else:
            raise ConfigError(
                f"Missing required configuration field '{f.name}' in {cls.__name__}"
            )

    return cls(**init_kwargs)


def load_decisions(path: Optional[Union[str, Path]] = None) -> DecisionsConfig:
    """Load decisions configuration with unknown key rejection.

    Args:
        path: Optional explicit path to decisions.yaml (default configs/decisions.yaml).

    Returns:
        Validated DecisionsConfig object.
    """
    cfg_path = path if path else "configs/decisions.yaml"
    raw_data = load_yaml(cfg_path)
    return bind_dataclass(DecisionsConfig, raw_data)


def load_config_with_decisions_override(
    base_config_path: Union[str, Path],
    decisions_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Load a base YAML configuration and merge with overrides from decisions.yaml."""
    base = load_yaml(base_config_path)
    decisions = load_yaml(decisions_path if decisions_path else "configs/decisions.yaml")

    merged = dict(base)
    # Apply matching keys from decisions if applicable
    for k, v in decisions.items():
        if k in merged:
            merged[k] = v
    return merged
