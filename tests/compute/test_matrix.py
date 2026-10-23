"""
tests/compute/test_matrix.py : Tests for experiment run matrix generation and plan unit totals.
"""

from src.rgpt.compute.matrix import generate_run_matrix


def test_run_matrix_counts_match_specs() -> None:
    """Matrix must contain 15 runs at 10M, 6 at 25M, 4 at 50M, and 2 gated at 100M."""
    matrix = generate_run_matrix()

    count_10m = sum(1 for r in matrix if r["stage"] == "pretrain_10m")
    count_25m = sum(1 for r in matrix if r["stage"] == "pretrain_25m")
    count_50m = sum(1 for r in matrix if r["stage"] == "pretrain_50m")
    count_100m = sum(1 for r in matrix if r["stage"] == "pretrain_100m")
    count_ec = sum(1 for r in matrix if "equal_compute" in r["stage"])
    count_bpeaks = sum(1 for r in matrix if "bpeaks" in r["stage"])

    assert count_10m == 15
    assert count_25m == 6
    assert count_50m == 4
    assert count_100m == 2
    assert count_ec == 6
    assert count_bpeaks == 5


def test_run_matrix_budget_reproduces_doc_plan() -> None:
    """Sum of planned units in run_matrix.csv must match 08_COMPUTE_BUDGET.md §6 totals within rounding."""
    matrix = generate_run_matrix()

    # 10M planned: 32.2 units (doc 08 §6)
    m10_units = sum(r["planned_units"] for r in matrix if r["stage"] == "pretrain_10m")
    assert abs(m10_units - 32.2) < 0.5

    # 25M planned: 52.2 + 2.24 ≈ 54.4 units
    m25_units = sum(r["planned_units"] for r in matrix if r["stage"] == "pretrain_25m")
    assert abs(m25_units - 54.4) < 0.5

    # 50M planned: 62.5 units
    m50_units = sum(r["planned_units"] for r in matrix if r["stage"] == "pretrain_50m")
    assert abs(m50_units - 62.5) < 0.5

    # 100M gated: 56.6 units (28.3 x 2)
    m100_units = sum(r["planned_units"] for r in matrix if r["stage"] == "pretrain_100m")
    assert abs(m100_units - 56.6) < 0.5
