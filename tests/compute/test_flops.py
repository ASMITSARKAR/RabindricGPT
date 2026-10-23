"""
tests/compute/test_flops.py : Tests verifying FLOPs and compute unit accounting formulas against doc 08 hand-computed anchors.
"""

import pytest

from src.rgpt.compute.flops import calculate_f_tok, calculate_run_flops, calculate_units


def test_flops_per_token_hand_anchors() -> None:
    """F_tok(10M, byte) must match 149.2M and F_tok(10M, bpe16) must match 108.5M to 3 sig digits."""
    # 10M model shape from doc 08 §4: L=6, d=384, N_ne = 10,616,832
    n_l = 6
    d = 384
    n_ne = 10616832

    # 10M Byte: V=264, s=3072 (doc 08 §5.1)
    # F_tok = 6*(10616832 + 264*384) + 12*6*384*3072 = 149,243,904 ≈ 149.2M
    f_byte = calculate_f_tok(n_ne=n_ne, vocab_size=264, d_model=d, n_layers=n_l, s_tokens=3072)
    assert pytest.approx(f_byte / 1e6, abs=0.1) == 149.2

    # 10M BPE16: V=16384, s=256 (doc 08 §5.2)
    # F_tok = 6*(10616832 + 16384*384) + 12*6*384*256 = 108,527,616 ≈ 108.5M
    f_bpe16 = calculate_f_tok(n_ne=n_ne, vocab_size=16384, d_model=d, n_layers=n_l, s_tokens=256)
    assert pytest.approx(f_bpe16 / 1e6, abs=0.1) == 108.5


def test_compute_units_hand_anchors() -> None:
    """units(10M byte) must match ~7.2 and units(10M bpe16) must match ~0.47 to 2 sig digits."""
    n_ne = 10616832
    d_bytes = 100 * n_ne  # 1.0616832e9

    # 10M Byte: T=1.0, FLOPs per unit = 2.2e16
    f_byte = calculate_f_tok(n_ne=n_ne, vocab_size=264, d_model=384, n_layers=6, s_tokens=3072)
    c_byte = calculate_run_flops(f_tok=f_byte, d_bytes=d_bytes, t_bytes_per_tok=1.0)
    units_byte = calculate_units(c_byte, flops_per_unit=2.2e16)
    assert pytest.approx(units_byte, abs=0.1) == 7.2

    # 10M BPE16: T=11.2, FLOPs per unit = 2.2e16
    f_bpe16 = calculate_f_tok(n_ne=n_ne, vocab_size=16384, d_model=384, n_layers=6, s_tokens=256)
    c_bpe16 = calculate_run_flops(f_tok=f_bpe16, d_bytes=d_bytes, t_bytes_per_tok=11.2)
    units_bpe16 = calculate_units(c_bpe16, flops_per_unit=2.2e16)
    assert pytest.approx(units_bpe16, abs=0.02) == 0.47
