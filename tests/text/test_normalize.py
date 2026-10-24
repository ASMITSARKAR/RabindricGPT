"""
tests/text/test_normalize.py : Tests for Bengali normalization, NFC, ZWJ/ZWNJ stripping policy,
Khanda-Ta canonicalization, U+09F0 mapping, and whitespace policy.
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


def test_zwj_zwnj_stripping_policy() -> None:
    """All ZWJ and ZWNJ must be stripped in the normalizer to prevent token fragmentation."""
    # Word-internal ZWJ / ZWNJ (e.g. ক্‍ষ, পোস্ট্‌মাস্টার)
    internal = "পোস্ট\u200cমাস্টার এবং ক\u09cd\u200dষ"
    norm_int, stats_int = normalize_text(internal)
    assert "\u200c" not in norm_int
    assert "\u200d" not in norm_int
    assert stats_int["zwnj_stripped"] == 1
    assert stats_int["zwj_stripped"] == 1
    assert stats_int["zwj_kept"] == 0
    assert stats_int["zwnj_kept"] == 0

    # Isolated / trailing ZWJ
    isolated = "হ্যালো \u200d বিশ্ব"
    norm_iso, stats_iso = normalize_text(isolated)
    assert "\u200d" not in norm_iso
    assert stats_iso["zwj_stripped"] == 1


def test_khanda_ta_canonicalization() -> None:
    """Historical ত্+ZWJ sequences must canonicalize to Khanda-Ta (ৎ, U+09CE).
    Explicit hasanta ত্+ZWNJ must NOT map to Khanda-Ta; stripping ZWNJ leaves visible hasanta ত্.
    """
    cases = [
        ("হঠাৎ", "হঠাৎ"),
        ("হত\u09cd\u200d", "হৎ"),          # ত্+ZWJ -> ৎ
        ("হত\u09cd\u200c", "হত\u09cd"),     # ত্+ZWNJ -> ত্ (visible hasanta, not ৎ)
        ("উত\u09cd\u200dসব", "উৎসব"),      # ত্+ZWJ -> ৎ
    ]
    for raw, expected in cases:
        norm, stats = normalize_text(raw)
        assert norm == expected, f"Failed canonicalization on {repr(raw)}: got {repr(norm)}"


def test_ryab_and_reph_normalization() -> None:
    """Ya-phala with ZWJ and reph are distinct forms and must normalize to different strings."""
    # 'র্যাব' with ZWJ: \u09b0 + \u09cd + \u200d + \u09af + \u09be + \u09ac
    ryab_with_zwj = "\u09b0\u09cd\u200d\u09af\u09be\u09ac"
    # 'র্যাব' without ZWJ (reph-ya): \u09b0 + \u09cd + \u09af + \u09be + \u09ac
    reph_ya = "\u09b0\u09cd\u09af\u09be\u09ac"

    norm_zwj, stats_zwj = normalize_text(ryab_with_zwj)
    norm_plain, stats_plain = normalize_text(reph_ya)

    # Must normalize to different strings
    assert norm_zwj != norm_plain
    assert norm_zwj == ryab_with_zwj
    assert norm_plain == reph_ya
    assert stats_zwj["zwj_kept"] == 1
    assert stats_zwj["zwj_stripped"] == 0

    # Other C+্+ZWJ+য forms also keep ZWJ
    ka_ya = "ক\u09cd\u200dযাব"
    norm_ka, stats_ka = normalize_text(ka_ya)
    assert norm_ka == ka_ya
    assert stats_ka["zwj_kept"] == 1

    # Alternative Avro order: C + ZWJ + virama + ya canonicalizes to C + virama + ZWJ + ya
    alt_ryab = "র\u200d\u09cd\u09afাব"
    norm_alt, _ = normalize_text(alt_ryab)
    assert norm_alt == ryab_with_zwj

    # Non-ya ZWJ is stripped
    dekhte = "দেখ\u200d্তে"
    norm_dekhte, stats_dekhte = normalize_text(dekhte)
    assert norm_dekhte == "দেখ্তে"
    assert stats_dekhte["zwj_stripped"] == 1



def test_u09f0_mapped_to_bengali_ra() -> None:
    """U+09F0 (Assamese Ra ৰ) must be mapped to Bengali Ra (র, U+09B0)."""
    text_assamese = "উৰ্দ্ধশ্বাসে ৰূপালী"
    norm, stats = normalize_text(text_assamese)
    assert "\u09f0" not in norm
    assert norm == "উর্দ্ধশ্বাসে রূপালী"
    assert stats["u09f0_mapped"] == 2


def test_poetry_newlines_preserved() -> None:
    """Single newlines in poetic stanzas must not be collapsed to spaces."""
    poem = "মেঘের কোলে রোদ হেসেছে,\nবাদল গেছে টুটি—\nআজ আমাদের ছুটি ও ভাই,\nআজ আমাদের ছুটি।"
    norm, _ = normalize_text(poem)
    assert norm.count("\n") == 3


def test_feff_bom_stripped() -> None:
    """U+FEFF (BOM / Zero-Width No-Break Space) must be stripped from text."""
    raw = "\ufeffরবীন্দ্রনাথ\ufeff \ufeffঠাকুর\ufeff\n"
    norm, stats = normalize_text(raw)
    assert "\ufeff" not in norm
    assert norm == "রবীন্দ্রনাথ ঠাকুর"
    assert stats["bom_stripped"] == 4
    assert stats["control_chars_removed"] >= 4

