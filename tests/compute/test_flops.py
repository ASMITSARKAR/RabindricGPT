"""
tests/compute/test_flops.py : Tests verifying FLOPs and compute unit accounting formulas against hand-computed constants for all 24 cells.
"""

import json
from pathlib import Path

import pytest
import yaml

from src.rgpt.compute.flops import calculate_f_tok, calculate_run_flops, calculate_units

# Hand-computed analytical FLOPs per token across all 24 cells from yaml L, d, V and measured s:
# F_tok = 6*(N_ne + V*d) + 12*L*d*s
HAND_CONSTANTS_ALL_24 = {
    ("10m", "byte"): 138819150.606484,
    ("10m", "char"): 93234676.119517,
    ("10m", "aks"): 101183414.601399,
    ("10m", "bpe8"): 91781249.057851,
    ("10m", "bpe16"): 109655332.455181,
    ("10m", "uni16"): 109594921.663064,
    ("25m", "byte"): 287412705.078194,
    ("25m", "char"): 206119687.323586,
    ("25m", "aks"): 212386457.958042,
    ("25m", "bpe8"): 195671496.991736,
    ("25m", "bpe16"): 219059037.253655,
    ("25m", "uni16"): 218951640.289892,
    ("50m", "byte"): 505352751.684679,
    ("50m", "char"): 378094331.443102,
    ("50m", "aks"): 380513360.559441,
    ("50m", "bpe8"): 354396234.049587,
    ("50m", "bpe16"): 383074945.708837,
    ("50m", "uni16"): 382907137.952956,
    ("100m", "byte"): 943465838.830261,
    ("100m", "char"): 729214579.224412,
    ("100m", "aks"): 719122772.139860,
    ("100m", "bpe8"): 675245999.603306,
    ("100m", "bpe16"): 708326740.790846,
    ("100m", "uni16"): 708044823.760966,
}

MODELS_SPEC = {
    "10m": {"L": 6, "d": 384, "N_ne": 10621824},
    "25m": {"L": 8, "d": 512, "N_ne": 25698816},
    "50m": {"L": 10, "d": 640, "N_ne": 49575040},
    "100m": {"L": 14, "d": 768, "N_ne": 99112704},
}

TOKENIZERS_SPEC = {
    "byte": {"V": 264},
    "char": {"V": 512},
    "aks": {"V": 8192},
    "bpe8": {"V": 8192},
    "bpe16": {"V": 16384},
    "uni16": {"V": 16384},
}


def test_flops_per_token_hand_constants_all_24_cells() -> None:
    """Every single one of the 24 cells must match its exact hand-computed constant from yaml L, d, V and measured s."""
    for (scale, tok), expected_f_tok in HAND_CONSTANTS_ALL_24.items():
        m_cfg = yaml.safe_load(Path(f"configs/models/{scale}.yaml").read_text(encoding="utf-8"))
        t_cfg = yaml.safe_load(Path(f"configs/tokenizers/{tok}.yaml").read_text(encoding="utf-8"))
        idx_p = Path(f"shards/wikisource/{tok}/train/index.json")
        meta = json.loads(idx_p.read_text(encoding="utf-8"))
        s_meas = meta["total_tokens"] / meta["total_chunks"]

        L = m_cfg["n_layers"]
        d = m_cfg["d_model"]
        V = t_cfg["vocab_size"]
        N_ne = MODELS_SPEC[scale]["N_ne"]

        computed = calculate_f_tok(
            n_ne=N_ne,
            vocab_size=V,
            d_model=d,
            n_layers=L,
            s_tokens=s_meas,
        )
        assert pytest.approx(computed, rel=1e-5) == expected_f_tok, (
            f"Cell ({scale}, {tok}): computed {computed} != expected {expected_f_tok}"
        )


def test_compute_units_equal_bytes_rule() -> None:
    """Verify compute units under the exact equal-bytes rule D_tok = D_bytes / T_meas.

    - 10M byte: D_bytes = 100 * 10,621,824. With T=1.0 B/tok, units match 6.70 within 0.02u.
    - 10M bpe16: on FineWeb (T=10.613 B/tok), units match 0.494 within 0.01u.
    - 10M bpe16: on Tagore (T=9.110 B/tok), units match 0.581 within 0.01u.
    """
    n_ne_10m = 10621824
    d_bytes = 100 * n_ne_10m

    # 1. 10M Byte (Tagore measured: s=2693.86, T_meas=1.000)
    f_byte_meas = calculate_f_tok(n_ne_10m, 264, 384, 6, 2694)
    c_byte = calculate_run_flops(f_tok=f_byte_meas, d_bytes=d_bytes, t_bytes_per_tok=1.0)
    units_byte = calculate_units(c_byte, flops_per_unit=2.2e16)
    assert pytest.approx(units_byte, abs=0.02) == 6.70

    # 2. 10M BPE16 Pretrain (FineWeb measured: s=254.53, T_meas=10.613)
    f_bpe16_fw = calculate_f_tok(n_ne_10m, 16384, 384, 6, 255)
    c_bpe16_fw = calculate_run_flops(f_tok=f_bpe16_fw, d_bytes=d_bytes, t_bytes_per_tok=10.613)
    units_bpe16_fw = calculate_units(c_bpe16_fw, flops_per_unit=2.2e16)
    assert pytest.approx(units_bpe16_fw, abs=0.01) == 0.494

    # 3. 10M BPE16 Adaptation (Tagore measured: s=295.71, T_meas=9.110)
    f_bpe16_tagore = calculate_f_tok(n_ne_10m, 16384, 384, 6, 296)
    c_bpe16_tagore = calculate_run_flops(f_tok=f_bpe16_tagore, d_bytes=d_bytes, t_bytes_per_tok=9.110)
    units_bpe16_tagore = calculate_units(c_bpe16_tagore, flops_per_unit=2.2e16)
    assert pytest.approx(units_bpe16_tagore, abs=0.01) == 0.581


def test_shard_sequence_length_drift_gate() -> None:
    """Test empirical shard sequence length s against planned s from configs.

    Asserts exact drift values on Tagore shards:
    - char (+0.04%), aks (+4.99%), bpe8 (+3.71%) PASS the 5% gate.
    - byte (-12.31%), bpe16 (+15.51%), uni16 (+14.66%) FAIL the 5% gate and are flagged.
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
        rel_dev = (s_meas - s_plan) / s_plan
        deviations[tid] = (s_plan, s_meas, rel_dev)

    # char, aks, and bpe8 pass the <= 5% gate
    assert abs(deviations["char"][2]) <= 0.05, f"char dev {deviations['char'][2]:.2%} > 5%"
    assert abs(deviations["aks"][2]) <= 0.05, f"aks dev {deviations['aks'][2]:.2%} > 5%"
    assert abs(deviations["bpe8"][2]) <= 0.05, f"bpe8 dev {deviations['bpe8'][2]:.2%} > 5%"

    # byte (-12.3%), bpe16 (+15.5%), and uni16 (+14.7%) exceed 5% gate and are flagged
    assert deviations["byte"][2] < -0.05, f"byte {deviations['byte'][2]:.2%} not flagged"
    assert deviations["bpe16"][2] > 0.05, f"bpe16 {deviations['bpe16'][2]:.2%} not flagged"
    assert deviations["uni16"][2] > 0.05, f"uni16 {deviations['uni16'][2]:.2%} not flagged"
