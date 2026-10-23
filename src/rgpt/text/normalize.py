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
        "control_chars_removed": 0,
    }

    # Step 1: NFC Normalization
    text = unicodedata.normalize("NFC", text)

    # Step 2: ZWJ / ZWNJ Policy
    # Keep inside Bengali words (surrounded by Bengali word characters), strip elsewhere
    out_chars = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch == ZWJ:
            prev_bengali = (i > 0 and text[i - 1] in BENGALI_WORD_CHARS)
            next_bengali = (i + 1 < n and text[i + 1] in BENGALI_WORD_CHARS)
            if prev_bengali and next_bengali:
                out_chars.append(ch)
                stats["zwj_kept"] += 1
            else:
                stats["zwj_stripped"] += 1
        elif ch == ZWNJ:
            prev_bengali = (i > 0 and text[i - 1] in BENGALI_WORD_CHARS)
            next_bengali = (i + 1 < n and text[i + 1] in BENGALI_WORD_CHARS)
            if prev_bengali and next_bengali:
                out_chars.append(ch)
                stats["zwnj_kept"] += 1
            else:
                stats["zwnj_stripped"] += 1
        elif unicodedata.category(ch).startswith("C") and ch not in ("\n", "\t"):
            # Control character removal
            stats["control_chars_removed"] += 1
        else:
            out_chars.append(ch)

    text = "".join(out_chars)

    # Step 3: Whitespace normalization
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
