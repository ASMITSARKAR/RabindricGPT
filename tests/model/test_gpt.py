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
