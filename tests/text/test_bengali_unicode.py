"""
tests/text/test_bengali_unicode.py : Tests for Unicode Bengali code-point sets and category uniqueness.
"""

from src.rgpt.text.bengali_unicode import (
    CONSONANTS,
    DANDAS,
    DIGITS,
    INDEPENDENT_VOWELS,
    MODIFIERS,
    VOWEL_SIGNS,
    classify_code_point,
    is_bengali_char,
)


def test_bengali_character_categories() -> None:
    """Every defined code point must match its Unicode Standard category."""
    # Consonants
    for ch in ["ক", "খ", "গ", "ঘ", "ড়", "ঢ়", "য়"]:
        assert classify_code_point(ch) == "CONSONANT"
        assert is_bengali_char(ch)

    # Independent Vowels
    for ch in ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ"]:
        assert classify_code_point(ch) == "INDEPENDENT_VOWEL"
        assert is_bengali_char(ch)

    # Modifiers
    for ch in ["ঁ", "ং", "ঃ"]:
        assert classify_code_point(ch) == "MODIFIER"
        assert is_bengali_char(ch)

    # Digits
    for ch in ["০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯"]:
        assert classify_code_point(ch) == "DIGIT"
        assert is_bengali_char(ch)

    # Dandas
    for ch in ["।", "॥"]:
        assert classify_code_point(ch) == "DANDA"
        assert is_bengali_char(ch)


def test_no_category_overlaps() -> None:
    """No code point may belong to multiple mutually exclusive grammatical categories."""
    categories = [
        CONSONANTS,
        INDEPENDENT_VOWELS,
        VOWEL_SIGNS,
        MODIFIERS,
        DIGITS,
        DANDAS,
    ]
    all_seen = set()
    for cat in categories:
        overlap = all_seen.intersection(cat)
        assert len(overlap) == 0, f"Overlapping code points found: {overlap}"
        all_seen.update(cat)
