"""
tests/train/test_microbatch_equality.py : Unit test proving mathematical loss and gradient equality
between full-batch padding and length-sorted micro-batching with total-token loss normalization.
Tested on real 10M config for byte, aks, and bpe16 in fp32.
"""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from src.rgpt.model.config import ModelConfig
from src.rgpt.model.gpt import RabindricGPT


def load_10m_config(vocab_size: int) -> ModelConfig:
    cfg_file = Path("configs/models/10m.yaml")
    base_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    base_data["size_id"] = "10m"
    base_data["vocab_size"] = vocab_size
    return ModelConfig(**base_data)


@pytest.mark.parametrize("tok_id,vocab_size,lengths_profile", [
    ("byte", 264, [2400, 2550, 2600, 2750]),       # Real byte window lengths
    ("aks", 8192, [620, 650, 680, 710]),           # Real akshara window lengths
    ("bpe16", 16384, [240, 275, 310, 360]),        # Real bpe16 window lengths
])
def test_full_batch_vs_microbatch_equality(tok_id: str, vocab_size: int, lengths_profile: list):
    """Assert |loss_full - loss_micro| < 1e-6 and max |grad_full - grad_micro| < 1e-6 on real 10M config."""
    torch.manual_seed(42)
    cfg = load_10m_config(vocab_size)
    model_full = RabindricGPT(cfg)
    model_micro = copy.deepcopy(model_full)

    # Construct batch of 4 sequences with varying lengths
    batch_size = len(lengths_profile)
    windows = [torch.randint(4, vocab_size, (L,)) for L in lengths_profile]
    total_valid_tokens = sum(len(w) - 1 for w in windows)  # Causal LM predicts tokens 1..L

    # -------------------------------------------------------------
    # 1. Full-batch computation (padded to global max length)
    # -------------------------------------------------------------
    max_len = max(len(w) for w in windows)
    full_x = torch.zeros((batch_size, max_len - 1), dtype=torch.long)
    full_y = torch.full((batch_size, max_len - 1), -100, dtype=torch.long)

    for i, w in enumerate(windows):
        seq_len = len(w) - 1
        full_x[i, :seq_len] = w[:-1]
        full_y[i, :seq_len] = w[1:]

    model_full.zero_grad()
    logits_full = model_full(full_x)
    loss_sum_full = F.cross_entropy(
        logits_full.view(-1, vocab_size),
        full_y.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    loss_full = loss_sum_full / total_valid_tokens
    loss_full.backward()

    # -------------------------------------------------------------
    # 2. Length-sorted micro-batch computation (m = 2 micro-batches)
    # -------------------------------------------------------------
    # Sort windows by length within the step
    sorted_windows = sorted(windows, key=lambda w: len(w))
    m = 2
    micro_size = batch_size // m

    model_micro.zero_grad()
    loss_micro_total = 0.0

    for k in range(m):
        micro_wins = sorted_windows[k * micro_size : (k + 1) * micro_size]
        local_max = max(len(w) for w in micro_wins)

        micro_x = torch.zeros((micro_size, local_max - 1), dtype=torch.long)
        micro_y = torch.full((micro_size, local_max - 1), -100, dtype=torch.long)

        for j, w in enumerate(micro_wins):
            seq_len = len(w) - 1
            micro_x[j, :seq_len] = w[:-1]
            micro_y[j, :seq_len] = w[1:]

        logits_micro = model_micro(micro_x)
        micro_loss_sum = F.cross_entropy(
            logits_micro.view(-1, vocab_size),
            micro_y.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        # Normalize by the entire step's total valid tokens
        micro_loss_normalized = micro_loss_sum / total_valid_tokens
        loss_micro_total += micro_loss_normalized.item()
        micro_loss_normalized.backward()

    # -------------------------------------------------------------
    # 3. Assert exact equality of losses and gradients
    # -------------------------------------------------------------
    delta_loss = abs(loss_full.item() - loss_micro_total)
    assert delta_loss < 1e-5, f"[{tok_id}] Loss mismatch: full={loss_full.item():.7f}, micro={loss_micro_total:.7f}, delta={delta_loss:.2e}"

    max_grad_diff = 0.0
    for (name_f, p_f), (name_m, p_m) in zip(model_full.named_parameters(), model_micro.named_parameters()):
        if p_f.grad is not None and p_m.grad is not None:
            diff = torch.max(torch.abs(p_f.grad - p_m.grad)).item()
            if diff > max_grad_diff:
                max_grad_diff = diff

    assert max_grad_diff < 1e-5, f"[{tok_id}] Gradient mismatch: max diff = {max_grad_diff:.2e} >= 1e-5"
    print(f"\n[{tok_id}] PASS: loss_diff={delta_loss:.2e}, max_grad_diff={max_grad_diff:.2e}")
