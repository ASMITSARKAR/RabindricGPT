"""
tests/analysis/test_hypotheses.py : Tests for automated evaluation of hypotheses H1–H6 against test scenarios.
"""

from src.rgpt.analysis.hypotheses import evaluate_hypotheses


def test_hypothesis_evaluation_decision_logic() -> None:
    """Hypothesis evaluator must correctly output CONFIRMED or REFUTED for known test matrices."""
    # Scenario: H1 confirmed, H2 confirmed, H3 confirmed, H4 confirmed, H5 confirmed, H6 confirmed
    confirmed_data = {
        "10m_bpc": {
            "bpe16": 1.25,
            "uni16": 1.26,
            "char": 1.50,
            "byte": 1.60,
            "akshara": 1.40,
        },
        "bpb_by_size": {
            "10m": {"akshara": 1.15, "bpe16": 1.05},
            "25m": {"akshara": 1.08, "bpe16": 1.00},
            "50m": {"akshara": 1.02, "bpe16": 0.96},
        },
        "akshara_vs_char_bpc": {
            "10m": (1.40, 1.50),
            "25m": (1.20, 1.30),
        },
        "equal_bytes_ranking": ["bpe16", "akshara", "char"],
        "equal_compute_ranking": ["bpe16", "char", "akshara"],
        "best_pretrain_tokenizer": "bpe16",
        "fastest_adapt_tokenizer": "bpe16",
        "byte_ill_formed_rate": 0.02,
        "subword_ill_formed_rate": 0.00,
    }

    verdicts = evaluate_hypotheses(confirmed_data)

    for h_id in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        assert h_id in verdicts
        assert verdicts[h_id]["verdict"] == "CONFIRMED"

    # Scenario: H1 refuted (byte beats bpe16)
    refuted_h1_data = {
        "10m_bpc": {
            "bpe16": 1.50,
            "byte": 1.40,
        }
    }
    v_refuted = evaluate_hypotheses(refuted_h1_data)
    assert v_refuted["H1"]["verdict"] == "REFUTED"
