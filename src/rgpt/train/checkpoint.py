"""
src/rgpt/train/checkpoint.py : Atomic checkpoint persistence, state recovery, and config-hash mismatch validation.
Backlog/Phase : I8 / Phase I      Docs : 15_BUILD_DIRECTIVE.md Phase I; 06_TRAINING_PROTOCOL.md; DR-14
Contract      : save_checkpoint(...) -> Path; load_checkpoint(...) -> dict
Invariants    : Atomic write-then-rename; keeps last K=2 plus final; refuses resume if config_hash differs
Failure modes : Config hash mismatch (hard-stops without --allow-config-change); corrupted checkpoint fallback
Status        : TESTED
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from src.rgpt.common.atomic_io import atomic_custom_write
from src.rgpt.common.seed import get_rng_states, set_rng_states


class CheckpointError(Exception):
    """Raised when checkpoint loading or verification fails."""
    pass


def save_checkpoint(
    checkpoint_dir: Union[str, Path],
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    config_hash: str,
    tokenizer_hash: str,
    keep_last: int = 2,
    is_final: bool = False,
) -> Path:
    """Save full training state atomically with atomic_custom_write."""
    cdir = Path(checkpoint_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    filename = "ckpt_final.pt" if is_final else f"ckpt_{step:08d}.pt"
    target_path = cdir / filename

    state = {
        "step": step,
        "config_hash": config_hash,
        "tokenizer_hash": tokenizer_hash,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_states": get_rng_states(),
    }

    def write_fn(temp_path: Path) -> None:
        torch.save(state, temp_path)

    atomic_custom_write(target_path, write_fn)

    # Prune older checkpoints keeping last `keep_last` non-final checkpoints
    if not is_final and keep_last > 0:
        pattern = re.compile(r"^ckpt_(\d{8})\.pt$")
        existing_ckpts = []
        for p in cdir.glob("ckpt_*.pt"):
            if p.name == "ckpt_final.pt":
                continue
            m = pattern.match(p.name)
            if m:
                existing_ckpts.append((int(m.group(1)), p))

        existing_ckpts.sort(key=lambda x: x[0])
        if len(existing_ckpts) > keep_last:
            to_remove = existing_ckpts[: -keep_last]
            for _, p_rm in to_remove:
                try:
                    p_rm.unlink()
                except OSError:
                    pass

    return target_path


def load_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    expected_config_hash: str,
    allow_config_change: bool = False,
) -> Optional[Dict[str, Any]]:
    """Discover and load the latest checkpoint, verifying config hash integrity."""
    cdir = Path(checkpoint_dir)
    if not cdir.exists():
        return None

    # Check for ckpt_final.pt or highest step ckpt_*.pt
    candidates = list(cdir.glob("ckpt_*.pt"))
    if not candidates:
        return None

    def get_ckpt_step(p: Path) -> int:
        if p.name == "ckpt_final.pt":
            return 999999999
        match = re.search(r"ckpt_(\d+)\.pt", p.name)
        return int(match.group(1)) if match else -1

    candidates.sort(key=get_ckpt_step, reverse=True)

    for ckpt_path in candidates:
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_cfg_hash = state.get("config_hash", "")

            if ckpt_cfg_hash != expected_config_hash and not allow_config_change:
                raise CheckpointError(
                    f"Config hash mismatch in checkpoint {ckpt_path.name}: "
                    f"expected '{expected_config_hash}', found '{ckpt_cfg_hash}'. "
                    f"Pass allow_config_change=True to override."
                )

            # Restore RNG states if present
            if "rng_states" in state:
                try:
                    set_rng_states(state["rng_states"])
                except Exception:
                    pass

            return state
        except CheckpointError:
            raise
        except Exception:
            # Corrupted checkpoint; fallback to next candidate
            continue

    return None
