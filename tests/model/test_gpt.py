"""
tests/model/test_gpt.py : Tests for RabindricGPT full architecture, weight tying, and forward shapes.
"""

import torch
import torch.nn.functional as F

from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import RabindricGPT


def test_tied_embeddings_invariant() -> None:
    """lm_head weight tensor must share storage identity with w_token embedding weight."""
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
    assert model.lm_head.weight is model.w_token.weight


def test_byte_window_shape_capacity() -> None:
    """Model must execute forward pass and return shape (B, T, V)."""
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
    )
    model = RabindricGPT(config)
    x = torch.randint(0, 264, (2, 64))
    logits = model(x)
    assert logits.shape == (2, 64, 264)


def test_overfit_single_batch() -> None:
    """Model must easily overfit a single training batch to near-zero loss."""
    torch.manual_seed(42)
    config = ModelConfig(
        size_id="tiny",
        n_layers=2,
        d_model=64,
        head_dim=32,
        n_heads=2,
        n_kv_heads=2,
        hidden_dim=128,
        vocab_size=100,
        max_seq_len=64,
    )
    model = RabindricGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.randint(0, 100, (1, 8))
    y = torch.randint(0, 100, (1, 8))

    initial_loss = float("inf")
    final_loss = float("inf")

    for step in range(35):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 100), y.view(-1))
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < 0.1
    assert final_loss < initial_loss / 10


def test_initial_loss_near_entropy_all_vocabs() -> None:
    """Item 3 / Level P2: |loss - ln(V)| < 0.10 across vocabs and model sizes on real configs."""
    import math
    from pathlib import Path

    import yaml

    vocabs = [264, 512, 8192, 16384]
    sizes = ["10m", "25m", "50m", "100m"]

    for sz in sizes:
        cfg_file = Path("configs/models") / f"{sz}.yaml"
        if not cfg_file.exists():
            continue
        base_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))

        for v in vocabs:
            cfg_dict = dict(base_data)
            cfg_dict["vocab_size"] = v
            cfg_dict["size_id"] = sz
            cfg = ModelConfig(**cfg_dict)

            torch.manual_seed(42)
            model = RabindricGPT(cfg)
            model.eval()

            # Random inputs and targets
            x = torch.randint(0, v, (8, 64))
            y = torch.randint(0, v, (8, 64))

            with torch.no_grad():
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, v), y.view(-1)).item()

            expected_loss = math.log(v)
            diff = abs(loss - expected_loss)
            assert diff < 0.10, f"Scale {sz}, Vocab {v}: init loss {loss:.4f} differs from ln(V) {expected_loss:.4f} by {diff:.4f} >= 0.10"

