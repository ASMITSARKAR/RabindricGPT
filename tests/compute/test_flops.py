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


def test_flops_rebuilt_tagore_shards_all_scales_strict_one_percent() -> None:
    """Read L, d from configs/models/*.yaml, V from configs/tokenizers/*.yaml,

    and s from Tagore train shard index per tokenizer; verify max deviation vs budget doc < 1%.
    """
    import json
    from pathlib import Path

    import yaml

    anchors = {
        "10m": {
            "char": calculate_f_tok(10616832, 512, 384, 6, 1024),
            "aks": calculate_f_tok(10616832, 8192, 384, 6, 640),
            "bpe16": calculate_f_tok(10616832, 16384, 384, 6, 256),
            "uni16": calculate_f_tok(10616832, 16384, 384, 6, 256),
            "bpe8": calculate_f_tok(10616832, 8192, 384, 6, 320),
        },
        "25m": {
            "char": calculate_f_tok(25165824, 512, 512, 8, 1024),
            "aks": calculate_f_tok(25165824, 8192, 512, 8, 640),
            "bpe16": calculate_f_tok(25165824, 16384, 512, 8, 256),
            "uni16": calculate_f_tok(25165824, 16384, 512, 8, 256),
            "bpe8": calculate_f_tok(25165824, 8192, 512, 8, 320),
        },
        "50m": {
            "char": calculate_f_tok(49152000, 512, 640, 10, 1024),
            "aks": calculate_f_tok(49152000, 8192, 640, 10, 640),
            "bpe16": calculate_f_tok(49152000, 16384, 640, 10, 256),
            "uni16": calculate_f_tok(49152000, 16384, 640, 10, 256),
            "bpe8": calculate_f_tok(49152000, 8192, 640, 10, 320),
        },
        "100m": {
            "char": calculate_f_tok(99090432, 512, 768, 14, 1024),
            "aks": calculate_f_tok(99090432, 8192, 768, 14, 640),
            "bpe16": calculate_f_tok(99090432, 16384, 768, 14, 256),
            "uni16": calculate_f_tok(99090432, 16384, 768, 14, 256),
            "bpe8": calculate_f_tok(99090432, 8192, 768, 14, 320),
        },
    }

    max_dev = 0.0
    for scale in ["10m", "25m", "50m", "100m"]:
        m_cfg = yaml.safe_load(Path(f"configs/models/{scale}.yaml").read_text(encoding="utf-8"))
        L = m_cfg["n_layers"]
        d = m_cfg["d_model"]
        n_ne = 12 * L * (d**2)
        for tid in ["char", "aks", "bpe8", "bpe16", "uni16"]:
            idx_p = Path(f"shards/wikisource/{tid}/train/index.json")
            if not idx_p.exists():
                continue
            meta = json.loads(idx_p.read_text(encoding="utf-8"))
            s_emp = int(meta["total_tokens"] // meta["total_chunks"])
            tok_cfg = yaml.safe_load(Path(f"configs/tokenizers/{tid}.yaml").read_text(encoding="utf-8"))
            V = tok_cfg["vocab_size"]

            f_emp = calculate_f_tok(n_ne=n_ne, vocab_size=V, d_model=d, n_layers=L, s_tokens=s_emp)
            target = anchors[scale][tid]
            rel_err = abs(f_emp - target) / target
            if rel_err > max_dev:
                max_dev = rel_err
            assert rel_err < 0.01, f"{scale} {tid} empirical FLOPs error {rel_err:.4%} exceeds 1%!"

    assert max_dev < 0.01, f"Max empirical FLOPs deviation {max_dev:.4%} exceeds 1%!"


