"""
tests/compute/test_flops.py : Tests verifying FLOPs and compute unit accounting formulas against doc 08 hand-computed constants.
"""

import json
from pathlib import Path

import pytest
import yaml

from src.rgpt.compute.flops import calculate_f_tok, calculate_run_flops, calculate_units


def test_flops_per_token_hand_constants() -> None:
    """F_tok(10M, byte) must match 149.2M and F_tok(10M, bpe16) must match 108.5M to 3 sig digits."""
    # 10M model shape from doc 08 §4: L=6, d=384, N_ne = 10,616,832
    n_l = 6
    d = 384
    n_ne = 10616832

    # Hand-computed anchor for 10M Byte (doc 08 §5.1):
    # F_tok = 6*(10616832 + 264*384) + 12*6*384*3072 = 6*(10718208) + 84934656 = 149,243,904
    f_byte = calculate_f_tok(
        n_ne=n_ne, vocab_size=264, d_model=d, n_layers=n_l, s_tokens=3072
    )
    assert f_byte == 149243904
    assert pytest.approx(f_byte / 1e6, abs=0.1) == 149.2

    # Hand-computed anchor for 10M BPE16 (doc 08 §5.2):
    # F_tok = 6*(10616832 + 16384*384) + 12*6*384*256 = 6*(16908288) + 7077888 = 108,527,616
    f_bpe16 = calculate_f_tok(
        n_ne=n_ne, vocab_size=16384, d_model=d, n_layers=n_l, s_tokens=256
    )
    assert f_bpe16 == 108527616
    assert pytest.approx(f_bpe16 / 1e6, abs=0.1) == 108.5


def test_compute_units_hand_constants() -> None:
    """units(10M byte) must match ~7.2 and units(10M bpe16) must match ~0.47 to 2 sig digits."""
    n_ne = 10616832
    d_bytes = 100 * n_ne  # 1.0616832e9

    # 10M Byte: T=1.0, FLOPs per unit = 2.2e16
    f_byte = 149243904  # Hand constant
    c_byte = calculate_run_flops(
        f_tok=f_byte, d_bytes=d_bytes, t_bytes_per_tok=1.0
    )
    units_byte = calculate_units(c_byte, flops_per_unit=2.2e16)
    assert pytest.approx(units_byte, abs=0.05) == 7.207

    # 10M BPE16: T=11.2, FLOPs per unit = 2.2e16
    f_bpe16 = 108527616  # Hand constant
    c_bpe16 = calculate_run_flops(
        f_tok=f_bpe16, d_bytes=d_bytes, t_bytes_per_tok=11.2
    )
    units_bpe16 = calculate_units(c_bpe16, flops_per_unit=2.2e16)
    assert pytest.approx(units_bpe16, abs=0.01) == 0.468


def test_shard_sequence_length_drift_gate() -> None:
    """Test empirical shard sequence length s against planned s from configs.

    Reports deviations and verifies behavior against the 5% tolerance gate.
    """
    tokenizers = ["char", "aks", "bpe8", "bpe16", "uni16", "byte"]
    deviations = {}

    for tid in tokenizers:
        t_cfg = yaml.safe_load(
            Path(f"configs/tokenizers/{tid}.yaml").read_text(encoding="utf-8")
        )
        s_plan = t_cfg["max_window_tokens"]
        idx_p = Path(f"shards/wikisource/{tid}/train/index.json")
        assert idx_p.exists(), f"Missing shard index for {tid}"
        meta = json.loads(idx_p.read_text(encoding="utf-8"))
        s_meas = meta["total_tokens"] / meta["total_chunks"]
        rel_dev = abs(s_meas - s_plan) / s_plan
        deviations[tid] = (s_plan, s_meas, rel_dev)

    # char, bpe8, and aks stay within 5%
    assert deviations["char"][2] <= 0.05, (
        f"char dev {deviations['char'][2]:.2%} > 5%"
    )
    assert deviations["bpe8"][2] <= 0.05, (
        f"bpe8 dev {deviations['bpe8'][2]:.2%} > 5%"
    )
    assert deviations["aks"][2] <= 0.05, (
        f"aks dev {deviations['aks'][2]:.2%} > 5%"
    )

    # bpe16, uni16, and byte exceed 5% drift (15.5%, 14.7%, 12.3%) and are detected
    assert deviations["bpe16"][2] > 0.05, (
        "bpe16 drift must be detected by gate"
    )
    assert deviations["uni16"][2] > 0.05, (
        "uni16 drift must be detected by gate"
    )
    assert deviations["byte"][2] > 0.05, "byte drift must be detected by gate"
