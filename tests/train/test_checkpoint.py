"""
tests/train/test_checkpoint.py : Tests for checkpoint persistence, atomic writes, and config-hash resume validation.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.rgpt.common.seed import seed_all
from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import RabindricGPT
from src.rgpt.train.checkpoint import (
    CheckpointError,
    load_latest_checkpoint,
    save_checkpoint,
)


def _make_toy_model():
    cfg = ModelConfig(
        size_id="toy",
        n_layers=2,
        d_model=64,
        head_dim=32,
        n_heads=2,
        n_kv_heads=2,
        hidden_dim=128,
        vocab_size=100,
        max_seq_len=64,
    )
    return RabindricGPT(cfg)


def test_resume_loss_equivalence(tmp_path: Path) -> None:
    """Training 40 steps uninterrupted must yield identical loss to training 20 steps + resume 20 steps."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()

    # Fixed seed and synthetic data
    seed_all(42)
    inputs = [torch.randint(0, 100, (2, 16)) for _ in range(40)]
    targets = [torch.randint(0, 100, (2, 16)) for _ in range(40)]

    # 1. Continuous 40-step run
    seed_all(42)
    model_cont = _make_toy_model()
    opt_cont = torch.optim.AdamW(model_cont.parameters(), lr=1e-3)

    losses_cont = []
    for step in range(40):
        opt_cont.zero_grad()
        out = model_cont(inputs[step])
        loss = F.cross_entropy(out.view(-1, 100), targets[step].view(-1))
        loss.backward()
        opt_cont.step()
        losses_cont.append(loss.item())

    # 2. Segment 1: Train 20 steps, save checkpoint
    seed_all(42)
    model_res = _make_toy_model()
    opt_res = torch.optim.AdamW(model_res.parameters(), lr=1e-3)

    for step in range(20):
        opt_res.zero_grad()
        out = model_res(inputs[step])
        loss = F.cross_entropy(out.view(-1, 100), targets[step].view(-1))
        loss.backward()
        opt_res.step()

    save_checkpoint(
        checkpoint_dir=ckpt_dir,
        step=19,
        model=model_res,
        optimizer=opt_res,
        scaler=None,
        config_hash="hash_a",
        tokenizer_hash="tok_a",
    )

    # 3. Segment 2: Resume from checkpoint and train steps 20..39
    loaded = load_latest_checkpoint(ckpt_dir, expected_config_hash="hash_a")
    assert loaded is not None
    assert loaded["step"] == 19

    model_res2 = _make_toy_model()
    opt_res2 = torch.optim.AdamW(model_res2.parameters(), lr=1e-3)
    model_res2.load_state_dict(loaded["model_state_dict"])
    opt_res2.load_state_dict(loaded["optimizer_state_dict"])

    losses_res = []
    for step in range(20, 40):
        opt_res2.zero_grad()
        out = model_res2(inputs[step])
        loss = F.cross_entropy(out.view(-1, 100), targets[step].view(-1))
        loss.backward()
        opt_res2.step()
        losses_res.append(loss.item())

    # Verify that loss trajectory in resumed run matches continuous run exactly
    for step_offset, (l_c, l_r) in enumerate(zip(losses_cont[20:], losses_res)):
        assert l_r == pytest.approx(l_c, rel=1e-4, abs=1e-4)


def test_config_hash_mismatch_refusal(tmp_path: Path) -> None:
    """Resuming from a checkpoint whose config hash differs must raise CheckpointError."""
    ckpt_dir = tmp_path / "ckpts_mismatch"
    ckpt_dir.mkdir()
    model = _make_toy_model()
    opt = torch.optim.AdamW(model.parameters())

    save_checkpoint(
        checkpoint_dir=ckpt_dir,
        step=5,
        model=model,
        optimizer=opt,
        scaler=None,
        config_hash="original_hash",
        tokenizer_hash="tok_a",
    )

    with pytest.raises(CheckpointError, match="Config hash mismatch"):
        load_latest_checkpoint(
            ckpt_dir,
            expected_config_hash="different_hash",
            allow_config_change=False,
        )


def test_corrupted_checkpoint_fallback(tmp_path: Path) -> None:
    """If the latest checkpoint is corrupted, loader must fall back to previous checkpoint."""
    ckpt_dir = tmp_path / "ckpts_corrupt"
    ckpt_dir.mkdir()
    model = _make_toy_model()
    opt = torch.optim.AdamW(model.parameters())

    save_checkpoint(ckpt_dir, step=1, model=model, optimizer=opt, scaler=None, config_hash="h", tokenizer_hash="t", keep_last=5)
    save_checkpoint(ckpt_dir, step=2, model=model, optimizer=opt, scaler=None, config_hash="h", tokenizer_hash="t", keep_last=5)

    # Corrupt step 2 checkpoint
    ckpt_2 = ckpt_dir / "ckpt_00000002.pt"
    ckpt_2.write_bytes(b"CORRUPTED_BYTES")

    # Loader must recover step 1
    loaded = load_latest_checkpoint(ckpt_dir, expected_config_hash="h")
    assert loaded is not None
    assert loaded["step"] == 1
