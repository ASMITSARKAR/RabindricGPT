"""
tests/data/test_leakage_audit.py : Tests verifying leakage plant sweep, script assertions, and 4-work detector.
"""

import pytest
from scripts.audits.audit_leakage_plants import (
    SUBSTITUTION_BANK_BENGALI,
    assert_bengali_script,
    generate_worst_case_indices,
)


def test_bengali_script_assertion() -> None:
    # Bengali words pass
    for w in SUBSTITUTION_BANK_BENGALI:
        assert_bengali_script(w)

    # Devanagari contamination fails
    # U+0937 (Devanagari SSA) + U+094D (Devanagari Virama)
    devanagari_word = "আবি\u0937\u094dকার"
    with pytest.raises(ValueError, match="Devanagari contamination detected"):
        assert_bengali_script(devanagari_word)


def test_worst_case_indices_arithmetic() -> None:
    # At length 13, 1 sub destroys at most 5 grams at index 6
    idxs_13_1 = generate_worst_case_indices(13, 1)
    assert idxs_13_1 == [6]

    # At length 13, 2 subs placed at 4 and 8
    idxs_13_2 = generate_worst_case_indices(13, 2)
    assert idxs_13_2 == [4, 8]
