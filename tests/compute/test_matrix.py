"""
tests/compute/test_matrix.py : Tests for experiment run matrix generation and plan unit totals.
"""

from src.rgpt.compute.matrix import generate_run_matrix


def test_run_matrix_counts_match_specs() -> None:
    """Revised Matrix: 14 runs at 10M, 9 at 25M, 3 at 50M, 0 approved at 100M, and 13 approved adaptation runs."""
    matrix = generate_run_matrix()

    count_10m = sum(1 for r in matrix if r["stage"] == "pretrain_10m" and r["approved"])
    count_25m = sum(1 for r in matrix if r["stage"] == "pretrain_25m" and r["approved"])
    count_50m = sum(1 for r in matrix if r["stage"] == "pretrain_50m" and r["approved"])
    count_100m_approved = sum(1 for r in matrix if r["stage"] == "pretrain_100m" and r["approved"])
    count_100m_unapproved = sum(1 for r in matrix if r["stage"] == "pretrain_100m" and not r["approved"])
    count_adapt_approved = sum(1 for r in matrix if "adapt" in r["stage"] and r["approved"])

    assert count_10m == 14  # 4x3 + 1x2 (byte)
    assert count_25m == 9   # 4x2 + 1x1 (byte)
    assert count_50m == 3   # aks, bpe16, uni16
    assert count_100m_approved == 0  # no 100M in approved funded plan
    assert count_100m_unapproved == 3  # 100M rows retained as unbudgeted / approved: False
    assert count_adapt_approved == 13  # 5 + 5 + 3


def test_run_matrix_budget_reproduces_doc_plan() -> None:
    """Sum of planned units must fit within owner cap of 200 units with 35-unit reserve (165 u ceiling)."""
    matrix = generate_run_matrix()

    # 10M: byte(2) + char(3) + aks(3) + bpe16(3) + uni16(3) = 25.08 u
    m10_units = sum(r["planned_units"] for r in matrix if r["stage"] == "pretrain_10m" and r["approved"])
    assert abs(m10_units - 25.08) < 0.5

    # 25M: byte(1) + char(2) + aks(2) + bpe16(2) + uni16(2) = 73.54 u
    m25_units = sum(r["planned_units"] for r in matrix if r["stage"] == "pretrain_25m" and r["approved"])
    assert abs(m25_units - 73.54) < 0.5

    # 50M: aks(1) + bpe16(1) + uni16(1) = 37.03 u
    m50_units = sum(r["planned_units"] for r in matrix if r["stage"] == "pretrain_50m" and r["approved"])
    assert abs(m50_units - 37.03) < 0.5

    # Adaptation runs: 13 approved runs on Tagore shards = ~0.93 u
    adapt_units = sum(r["planned_units"] for r in matrix if "adapt" in r["stage"] and r["approved"])
    assert abs(adapt_units - 0.93) < 0.2

    # Total spend must stay under 165 units (200 cap - 35 reserve)
    total_units = sum(r["planned_units"] for r in matrix if r["approved"])
    assert total_units <= 165.0
    assert abs(total_units - 136.58) < 1.0
    assert abs(total_units - 136.58) < 1.0
