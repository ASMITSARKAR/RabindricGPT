"""
src/rgpt/model/config.py : Model architecture configuration dataclass with validation and sizing methods.
Backlog/Phase : I7 / Phase H      Docs : 15_BUILD_DIRECTIVE.md Phase H; 05_MODEL_SPEC.md; DR-3
Contract      : yaml_path: Path -> ModelConfig; ModelConfig.to_dict() -> dict
Invariants    : n_heads = d_model / 64; hidden_dim is multiple of 64; tie_embeddings is True
Failure modes : Invalid dimensions (d_model not divisible by 64); unsupported head count
Status        : TESTED
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for RabindricGPT decoder-only architecture."""

    size_id: str
    n_layers: int
    d_model: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    hidden_dim: int
    vocab_size: int
    max_seq_len: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    embedding_init_scale: float = 0.35

    def __post_init__(self) -> None:
        if self.d_model % self.head_dim != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by head_dim ({self.head_dim})"
            )
        expected_heads = self.d_model // self.head_dim
        if self.n_heads != expected_heads:
            raise ValueError(
                f"n_heads ({self.n_heads}) must equal d_model // head_dim ({expected_heads})"
            )
        if self.hidden_dim % 64 != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be a multiple of 64 per DR-3"
            )

    @classmethod
    def from_yaml(cls, path: Union[str, Path], vocab_size: int) -> "ModelConfig":
        """Load ModelConfig from YAML file and inject vocabulary size."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model config file not found: {p}")

        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return cls(
            size_id=data["size_id"],
            n_layers=int(data["n_layers"]),
            d_model=int(data["d_model"]),
            head_dim=int(data.get("head_dim", 64)),
            n_heads=int(data["n_heads"]),
            n_kv_heads=int(data.get("n_kv_heads", data["n_heads"])),
            hidden_dim=int(data["hidden_dim"]),
            vocab_size=int(vocab_size),
            max_seq_len=int(data.get("max_seq_len", 3072)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-5)),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            tie_embeddings=bool(data.get("tie_embeddings", True)),
            embedding_init_scale=float(data.get("embedding_init_scale", 0.35)),
        )

    @classmethod
    def from_size(cls, size_id: str, vocab_size: int) -> "ModelConfig":
        """Load ModelConfig by size ID ('10m', '25m', '50m', '100m')."""
        from src.rgpt.common.paths import get_project_root
        yaml_p = get_project_root() / "configs" / "models" / f"{size_id}.yaml"
        return cls.from_yaml(yaml_p, vocab_size)


    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
