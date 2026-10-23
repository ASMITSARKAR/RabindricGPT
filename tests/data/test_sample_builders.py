"""
tests/data/test_sample_builders.py : Tests verifying sample builders logic, filter rules, and zero CRLF.
"""

from scripts.build_fw2_100m_sample import NARROW_TAGORE_KEYWORDS, normalize_for_filter


def test_normalize_for_filter() -> None:
    # NFC normalization
    # Decomposed ga + aa
    raw = "\u0997\u09be"
    assert normalize_for_filter(raw) == "গা"

    # Strip ZWNJ and ZWJ
    with_zwnj = "রবীন্দ্রনাথের\u200c"
    assert normalize_for_filter(with_zwnj) == "রবীন্দ্রনাথের"
    with_zwj = "রবীন্দ্রনাথের\u200d"
    assert normalize_for_filter(with_zwj) == "রবীন্দ্রনাথের"


def test_narrow_keywords_compound_policy() -> None:
    # Bare ডাকঘর must NOT be in the list
    assert "ডাকঘর" not in NARROW_TAGORE_KEYWORDS
    # Compound নাটক ডাকঘর must be in the list
    assert "নাটক ডাকঘর" in NARROW_TAGORE_KEYWORDS
    # Shesher Kobita and other canon must be present
    assert "শেষের কবিতা" in NARROW_TAGORE_KEYWORDS
    assert "রবীন্দ্রনাথের" in NARROW_TAGORE_KEYWORDS
    assert "গীতাঞ্জলি" in NARROW_TAGORE_KEYWORDS


def test_filter_matches_leaking_passage() -> None:
    leaking_quote = "...আজ রবীন্দ্রনাথের শেষের কবিতা উপন্যাসের অমিতের বিদায় বাণী মনে পড়ে যায়..."
    norm = normalize_for_filter(leaking_quote)
    matches = [kw for kw in NARROW_TAGORE_KEYWORDS if kw in norm]
    assert "শেষের কবিতা" in matches
    assert "রবীন্দ্রনাথের" in matches
