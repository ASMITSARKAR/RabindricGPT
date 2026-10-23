"""
tests/text/test_normalize.py : Tests for Bengali normalization, NFC, ZWJ/ZWNJ preservation, and whitespace policy.
"""

from src.rgpt.text.normalize import is_normalized, normalize_text


def test_normalization_idempotent() -> None:
    """Repeated normalization must not alter text: norm(norm(x)) == norm(x)."""
    samples = [
        "সোনার তরী— রবীন্দ্রনাথ ঠাকুর",
        "ক\u09bc \u09cd\u200d র\u09cd\u200cয",
        "লাইন ১\n\n\nলাইন ২    অনেক স্পেস।",
    ]
    for s in samples:
        norm1, _ = normalize_text(s)
        norm2, _ = normalize_text(norm1)
        assert norm1 == norm2
        assert is_normalized(norm1)


def test_nukta_composition_invariance() -> None:
    """Nukta forms must remain base + U+09BC under NFC without decomposition surprises."""
    # ড় can be represented as U+09DC or ড (U+09A1) + nukta (U+09BC)
    text = "বাড়ি"
    norm, _ = normalize_text(text)
    # Under NFC in Python unicodedata, \u09dc stays \u09dc (or \u09a1\u09bc), both are valid NFC
    assert "বাড়ি" in norm or "বাড\u09bcি" in norm


def test_zwj_zwnj_preservation_and_stripping() -> None:
    """ZWJ/ZWNJ inside Bengali words must be kept; orphan ZWJ/ZWNJ stripped."""
    # Word-internal ZWJ / ZWNJ (e.g. ক্‍ষ)
    internal = "ক\u09cd\u200dষ"
    norm_int, stats_int = normalize_text(internal)
    assert "\u200d" in norm_int
    assert stats_int["zwj_kept"] == 1

    # Isolated / trailing ZWJ
    isolated = "হ্যালো \u200d বিশ্ব"
    norm_iso, stats_iso = normalize_text(isolated)
    assert "\u200d" not in norm_iso
    assert stats_iso["zwj_stripped"] == 1


def test_poetry_newlines_preserved() -> None:
    """Single newlines in poetic stanzas must not be collapsed to spaces."""
    poem = "মেঘের কোলে রোদ হেসেছে,\nবাদল গেছে টুটি—\nআজ আমাদের ছুটি ও ভাই,\nআজ আমাদের ছুটি।"
    norm, _ = normalize_text(poem)
    assert norm.count("\n") == 3
