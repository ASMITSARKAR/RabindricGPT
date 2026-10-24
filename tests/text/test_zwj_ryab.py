"""Unit tests specifically verifying ZWJ ya-phala vs reph normalization,
code points, and distinct string representations.
"""

import sys
from src.rgpt.text.normalize import normalize_text
from src.rgpt.text.akshara import segment

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def format_codepoints(s: str) -> str:
    return " ".join(f"U+{ord(c):04X}" for c in s)


def test_ryab_distinct_codepoints_and_normalization() -> None:
    # 1. Ya-phala form with ZWJ: র + virama + ZWJ + ya + aa + ba
    # Code points: U+09B0 U+09CD U+200D U+09AF U+09BE U+09AC
    ryab_with_zwj = "\u09b0\u09cd\u200d\u09af\u09be\u09ac"

    # 2. Reph-ya form without ZWJ: র + virama + ya + aa + ba
    # Code points: U+09B0 U+09CD U+09AF U+09BE U+09AC
    ryab_plain_reph = "\u09b0\u09cd\u09af\u09be\u09ac"

    # 3. Legacy Avro order: র + ZWJ + virama + ya + aa + ba
    # Code points: U+09B0 U+200D U+09CD U+09AF U+09BE U+09AC
    ryab_legacy_avro = "\u09b0\u200d\u09cd\u09af\u09be\u09ac"

    # Verify initial raw code points are different
    cp_zwj = format_codepoints(ryab_with_zwj)
    cp_plain = format_codepoints(ryab_plain_reph)
    cp_legacy = format_codepoints(ryab_legacy_avro)

    print("\n--- RAW INPUT STRINGS ---")
    print(f"1. ryab_with_zwj   : Code points: {cp_zwj}")
    print(f"2. ryab_plain_reph : Code points: {cp_plain}")
    print(f"3. ryab_legacy_avro: Code points: {cp_legacy}")

    assert ryab_with_zwj != ryab_plain_reph
    assert len(ryab_with_zwj) == 6
    assert len(ryab_plain_reph) == 5
    assert len(ryab_legacy_avro) == 6

    # Normalize each string
    norm_zwj, stats_zwj = normalize_text(ryab_with_zwj)
    norm_plain, stats_plain = normalize_text(ryab_plain_reph)
    norm_legacy, stats_legacy = normalize_text(ryab_legacy_avro)

    cp_norm_zwj = format_codepoints(norm_zwj)
    cp_norm_plain = format_codepoints(norm_plain)
    cp_norm_legacy = format_codepoints(norm_legacy)

    print("\n--- NORMALIZED OUTPUT STRINGS ---")
    print(f"1. norm_zwj   : Code points: {cp_norm_zwj}")
    print(f"2. norm_plain : Code points: {cp_norm_plain}")
    print(f"3. norm_legacy: Code points: {cp_norm_legacy}")

    # Assertions
    # 1. ZWJ ya-phala remains distinct from plain reph-ya
    assert norm_zwj != norm_plain, f"norm_zwj ({cp_norm_zwj}) must not equal norm_plain ({cp_norm_plain})"
    assert "\u200d" in norm_zwj
    assert "\u200d" not in norm_plain
    assert stats_zwj["zwj_kept"] == 1
    assert stats_zwj["zwj_stripped"] == 0

    # 2. Legacy Avro order canonicalizes to standard ZWJ ya-phala
    assert norm_legacy == norm_zwj, f"Legacy order {cp_norm_legacy} must canonicalize to {cp_norm_zwj}"
    assert stats_legacy["zwj_kept"] == 1


def test_akshara_segmenter_keeps_c_virama_zwj_ya() -> None:
    ryab_with_zwj = "\u09b0\u09cd\u200d\u09af\u09be\u09ac"
    units = segment(ryab_with_zwj)
    print(f"\nAkshara segmentation of ryab_with_zwj: {len(units)} units")
    for idx, u in enumerate(units):
        print(f"  Unit {idx+1}: Code points: {format_codepoints(u)}")

    # First unit must be 'র‍্যা' (U+09B0 U+09CD U+200D U+09AF U+09BE)
    # Second unit must be 'ব' (U+09AC)
    assert len(units) == 2
    assert units[0] == "\u09b0\u09cd\u200d\u09af\u09be"
    assert units[1] == "\u09ac"
    assert "".join(units) == ryab_with_zwj
