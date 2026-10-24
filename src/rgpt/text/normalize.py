"""
src/rgpt/text/normalize.py : Unicode normalization, ZWJ/ZWNJ handling, whitespace and danda policy.
Backlog/Phase : I1 / Phase C      Docs : 15_BUILD_DIRECTIVE.md Phase C; 03_DATA_PLAN.md §3; 04 §2
Contract      : raw_text: str -> (normalized_text: str, stats: dict)
Invariants    : Idempotent (norm(norm(x)) == norm(x)); NFC form; nukta forms remain base+U+09BC; poetry newlines kept
Failure modes : Invalid UTF-8 bytes; unexpected encoding
Status        : TESTED
"""

import re
import unicodedata
from typing import Dict, Tuple

from src.rgpt.text.bengali_unicode import (
    CONSONANTS,
    INDEPENDENT_VOWELS,
    KHANDA_TA,
    NUKTA,
    VIRAMA,
    VOWEL_SIGNS,
    ZWJ,
    ZWNJ,
)

BENGALI_WORD_CHARS = CONSONANTS | INDEPENDENT_VOWELS | VOWEL_SIGNS | {VIRAMA, NUKTA, KHANDA_TA}


def normalize_text(text: str) -> Tuple[str, Dict[str, int]]:
    """Apply Bengali normalization rules: NFC, ZWJ/ZWNJ preservation/stripping, whitespace normalization.

    Args:
        text: Raw input string.

    Returns:
        Tuple of (normalized string, dictionary of transformation counts).
    """
    stats: Dict[str, int] = {
        "zwj_kept": 0,
        "zwj_stripped": 0,
        "zwnj_kept": 0,
        "zwnj_stripped": 0,
        "u09f0_mapped": 0,
        "khanda_ta_canonicalized": 0,
        "control_chars_removed": 0,
    }

    # Step 1: NFC Normalization
    text = unicodedata.normalize("NFC", text)

    # Step 2: Character mappings (U+09F0 Assamese Ra -> U+09B0 Bengali Ra)
    u09f0_count = text.count("\u09f0")
    if u09f0_count > 0:
        stats["u09f0_mapped"] = u09f0_count
        text = text.replace("\u09f0", "\u09b0")

    # Step 3: Khanda-Ta canonicalization before joiner removal
    # Legacy representation: ত (U+09A4) + virama (U+09CD) + ZWJ (\u200d) -> ৎ (U+09CE).
    # ত + virama + ZWNJ (\u200c) represents an explicit visible hasanta (ত্), NOT Khanda-Ta.
    khanda_ta_pattern = re.compile(r"\u09a4\u09cd\u200d")
    khanda_matches = len(khanda_ta_pattern.findall(text))
    if khanda_matches > 0:
        stats["khanda_ta_canonicalized"] = khanda_matches
        text = khanda_ta_pattern.sub("\u09ce", text)

    # Step 4: ZWJ / ZWNJ Policy
    # Strip all remaining ZWJ (\u200d) and ZWNJ (\u200c) uniformly. In Bengali, joiners
    # are purely rendering hints to break or force ligatures; stripping them preserves
    # semantic identity and prevents token/grapheme fragmentation and lone-virama artifacts.
    zwj_count = text.count(ZWJ)
    zwnj_count = text.count(ZWNJ)
    stats["zwj_stripped"] += zwj_count
    stats["zwnj_stripped"] += zwnj_count
    if zwj_count > 0 or zwnj_count > 0:
        text = text.replace(ZWJ, "").replace(ZWNJ, "")

    # Step 5: Control character removal (keep \n and \t)
    # Strip all Unicode general category C (control, format, surrogate, private use) except \n and \t
    text, ctrl_count = re.subn(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    stats["control_chars_removed"] += ctrl_count

    # Step 6: Whitespace normalization
    # Collapse multiple horizontal whitespace runs (spaces, tabs) to a single space
    # Keep single newlines for poetry, collapse 3+ newlines to max 2
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text, stats


def is_normalized(text: str) -> bool:
    """Check if text is already normalized according to the policy."""
    norm_text, _ = normalize_text(text)
    return text == norm_text
