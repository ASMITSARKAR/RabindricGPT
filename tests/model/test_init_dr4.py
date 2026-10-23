"""
tests/model/test_init_dr4.py : Automated unit tests for DR-4 parameter initialization,
per-module standard deviations, and |loss - ln(V)| < 0.10 bound across all scales and vocabs.
"""

import math
from pathlib import Path

import pytest
import torch
import yaml

from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import RabindricGPT


@pytest.mark.parametrize("size_id", ["10m", "25m", "50m", "100m"])
def test_per_module_std_matches_spec(size_id: str):
    """Test that hidden linears, residual projections, and embeddings have exact target stds."""
    cfg_file = Path("configs/models") / f"{size_id}.yaml"
    base_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    base_data["size_id"] = size_id
    base_data["vocab_size"] = 8192
    cfg = ModelConfig(**base_data)
    model = RabindricGPT(cfg)

    # 1. Embedding std target: embedding_init_scale / sqrt(d)
    expected_emb_std = cfg.embedding_init_scale / (cfg.d_model**0.5)
    measured_emb_std = model.w_token.weight.data.std().item()
    assert abs(measured_emb_std - expected_emb_std) < 0.003, (
        f"Scale {size_id}: Embedding std {measured_emb_std:.5f} differs from target {expected_emb_std:.5f}"
    )

    # 2. Hidden linear std target: 0.02 (DR-4 canonical standard)
    for block in model.blocks:
        q_std = block.attn.q_proj.weight.data.std().item()
        assert abs(q_std - 0.02) < 0.003, f"Scale {size_id}: Hidden linear std {q_std:.5f} differs from DR-4 0.02"

        gate_std = block.mlp.w_gate.weight.data.std().item()
        assert abs(gate_std - 0.02) < 0.003, f"Scale {size_id}: MLP gate std {gate_std:.5f} differs from DR-4 0.02"

    # 3. Residual projection std target: 0.02 / sqrt(2L) per DR-4
    expected_res_std = 0.02 / ((2.0 * cfg.n_layers)**0.5)
    for block in model.blocks:
        out_std = block.attn.out_proj.weight.data.std().item()
        assert abs(out_std - expected_res_std) < 0.002, (
            f"Scale {size_id}: Residual out_proj std {out_std:.5f} differs from target {expected_res_std:.5f}"
        )
        down_std = block.mlp.w_down.weight.data.std().item()
        assert abs(down_std - expected_res_std) < 0.002, (
            f"Scale {size_id}: Residual mlp down std {down_std:.5f} differs from target {expected_res_std:.5f}"
        )


@pytest.mark.parametrize("size_id", ["10m", "25m", "50m", "100m"])
@pytest.mark.parametrize("vocab_size", [264, 512, 8192, 16384])
def test_initial_loss_near_entropy_bound(size_id: str, vocab_size: int):
    """Prove |loss - ln(V)| < 0.10 across all 16 scale x vocab combinations."""
    torch.manual_seed(42)
    cfg_file = Path("configs/models") / f"{size_id}.yaml"
    base_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    base_data["size_id"] = size_id
    base_data["vocab_size"] = vocab_size
    cfg = ModelConfig(**base_data)
    model = RabindricGPT(cfg)
    model.eval()

    # Draw representative batch
    x = torch.randint(0, vocab_size, (8, 64))
    y = torch.randint(0, vocab_size, (8, 64))

    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size), y.view(-1)
        ).item()

    ln_v = math.log(vocab_size)
    delta = abs(loss - ln_v)
    assert delta < 0.10, (
        f"Scale {size_id} (d={cfg.d_model}, L={cfg.n_layers}), V={vocab_size}: "
        f"loss={loss:.4f}, ln(V)={ln_v:.4f}, |loss - ln(V)| = {delta:.4f} >= 0.10!"
    )
