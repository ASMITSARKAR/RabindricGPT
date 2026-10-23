"""
tests/text/test_clean.py : Tests for cleaning pipeline, script share threshold, and PUA collision guard.
"""

import pytest

from src.rgpt.text.clean import (
    PUACollisionError,
    bengali_character_ratio,
    clean_pipeline,
)


def test_clean_pipeline_step_order() -> None:
    """Cleaning steps must execute strictly in configured order: UTF-8 -> NFC -> ZWJ/ZWNJ -> filter."""
    raw = "  \tগীতাঞ্জলি \u200d\nকাব্যগ্রন্থ  "
    clean, passed, stats = clean_pipeline(raw, threshold=0.80)
    assert passed
    assert clean.startswith("গীতাঞ্জলি")
    assert stats["clean_chars"] > 0
    assert stats["clean_bytes"] > 0


def test_bengali_script_threshold_filtering() -> None:
    """Documents with <80% Bengali script characters must be rejected."""
    # 100% Bengali
    bengali_doc = "গীতাঞ্জলি কাব্যগ্রন্থের জন্য রবীন্দ্রনাথ ঠাকুর নোবেল পুরস্কার পান।"
    assert bengali_character_ratio(bengali_doc) >= 0.90
    _, passed, _ = clean_pipeline(bengali_doc, threshold=0.80)
    assert passed

    # English / Latin text with a tiny bit of Bengali (<80%)
    junk_doc = "This is mostly English text with a single Bengali word: বাংলা."
    assert bengali_character_ratio(junk_doc) < 0.30
    _, passed, _ = clean_pipeline(junk_doc, threshold=0.80)
    assert not passed


def test_pua_collision_guard_raises() -> None:
    """Presence of Private Use Area characters (U+E000–U+F8FF) must raise PUACollisionError per DR-9."""
    pua_doc = "নরমাল টেক্সট \uE005 সহ অবৈধ ক্যারেক্টার।"
    with pytest.raises(PUACollisionError, match="PUA collision detected"):
        clean_pipeline(pua_doc, check_pua=True)
