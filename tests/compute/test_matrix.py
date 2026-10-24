"""
tests/compute/test_matrix.py : Tests for experiment run matrix generation and plan unit totals.
"""

from src.rgpt.compute.matrix import generate_run_matrix


def test_run_matrix_counts_match_specs() -> None:
    """Approved list: 10M all 5 (14 runs), 25M all 5 (1 seed each = 5 runs), 50M aks and bpe16 (2 runs), 12 adapt runs."""
    matrix = generate_run_matrix()

    count_10m = sum(1 for r in matrix if r["stage"] == "pretrain_10m" and r["approved"])
    count_25m = sum(1 for r in matrix if r["stage"] == "pretrain_25m" and r["approved"])
    count_50m = sum(1 for r in matrix if r["stage"] == "pretrain_50m" and r["approved"])
    count_100m_approved = sum(1 for r in matrix if r["stage"] == "pretrain_100m" and r["approved"])
    count_100m_unapproved = sum(1 for r in matrix if r["stage"] == "pretrain_100m" and not r["approved"])
    count_adapt_approved = sum(1 for r in matrix if "adapt" in r["stage"] and r["approved"])

    assert count_10m == 14  # 4x3 + 1x2 (byte)
    assert count_25m == 5   # 5x1 (byte, char, aks, bpe16, uni16 - seed 0)
    assert count_50m == 2   # aks, bpe16 (1 seed each)
    assert count_100m_approved == 0  # no 100M in approved funded plan
    assert count_100m_unapproved == 3  # 100M rows retained as unbudgeted / approved: False
    assert count_adapt_approved == 12  # 5 + 5 + 2


def test_run_matrix_budget_reproduces_doc_plan() -> None:
    """Sum of planned units must confirm <= 150 u ceiling at empirical rates."""
    matrix = generate_run_matrix()

    # 10M: byte(2) + char(3) + aks(3) + bpe16(3) + uni16(3) = 49.36 u
    m10_units = sum(r["planned_units"] for r in matrix if r["stage"] == "pretrain_10m" and r["approved"])
    assert abs(m10_units - 49.36) < 0.5

    # 25M: byte(1) + char(1) + aks(1) + bpe16(1) + uni16(1) = 49.37 u
    m25_units = sum(r["planned_units"] for r in matrix if r["stage"] == "pretrain_25m" and r["approved"])
    assert abs(m25_units - 49.37) < 0.5

    # 50M: aks(1) + bpe16(1) = 17.19 u
    m50_units = sum(r["planned_units"] for r in matrix if r["stage"] == "pretrain_50m" and r["approved"])
    assert abs(m50_units - 17.19) < 0.5

    # Adaptation runs: 12 approved runs on Tagore shards = 1.17 u
    adapt_units = sum(r["planned_units"] for r in matrix if "adapt" in r["stage"] and r["approved"])
    assert abs(adapt_units - 1.17) < 0.5

    # Total spend must stay under 150 units (200 cap - 35 reserve = 165 ceiling, user confirmed <= 150 u)
    total_units = sum(r["planned_units"] for r in matrix if r["approved"])
    assert total_units <= 150.0
    assert abs(total_units - 117.09) < 0.5
