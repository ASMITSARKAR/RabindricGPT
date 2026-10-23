"""
src/rgpt/text/clean.py : Ordered text cleaning pipeline, script share filtering, and PUA safety checks.
Backlog/Phase : I1 / Phase C      Docs : 15_BUILD_DIRECTIVE.md Phase C; 03_DATA_PLAN.md §3; DR-9
Contract      : text: str -> (cleaned_text: str, passed_filter: bool, flags: dict)
Invariants    : Pipeline order is strictly UTF-8 -> NFC -> ZWJ/ZWNJ -> control/whitespace -> filter -> PUA check
Failure modes : PUA character found (raises loud error per DR-9); Bengali share < threshold
Status        : TESTED
"""

from typing import Any, Dict, Tuple

from src.rgpt.text.bengali_unicode import is_bengali_char
from src.rgpt.text.normalize import normalize_text


class PUACollisionError(Exception):
    """Raised when a Private Use Area (PUA) code point is detected in cleaned text."""
    pass


def check_pua_collision(text: str) -> None:
    """Raise PUACollisionError if text contains any code point in U+E000–U+F8FF."""
    for ch in text:
        cp = ord(ch)
        if 0xE000 <= cp <= 0xF8FF:
            raise PUACollisionError(
                f"PUA collision detected: character U+{cp:04X} found in text. DR-9 forbids PUA in cleaned text."
            )


def bengali_character_ratio(text: str) -> float:
    """Compute the fraction of non-whitespace characters that belong to the Bengali script.

    Args:
        text: Input text string.

    Returns:
        Fraction in [0.0, 1.0].
    """
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 0.0

    bengali_count = sum(1 for ch in non_space if is_bengali_char(ch))
    return bengali_count / len(non_space)


def clean_pipeline(
    text: str,
    threshold: float = 0.80,
    check_pua: bool = True
) -> Tuple[str, bool, Dict[str, Any]]:
    """Execute the full cleaning and filtering pipeline on a document.

    Order:
        1. UTF-8 decode (ensured by caller passing str)
        2. Normalize (NFC, ZWJ/ZWNJ preservation, control chars, whitespace)
        3. Script ratio threshold check (default >=80% Bengali)
        4. PUA guard (raises PUACollisionError if U+E000-U+F8FF present)

    Args:
        text: Input raw document string.
        threshold: Minimum required Bengali script share (default 0.80).
        check_pua: If True, raise PUACollisionError if U+E000–U+F8FF appears.

    Returns:
        Tuple of (cleaned text, pass_filter_boolean, detailed stats).
    """
    if check_pua:
        check_pua_collision(text)

    # Step 2: Normalization
    norm_text, norm_stats = normalize_text(text)

    # Step 3: Script share calculation
    ratio = bengali_character_ratio(norm_text)
    passed_filter = (ratio >= threshold)

    stats: Dict[str, Any] = {
        "raw_chars": len(text),
        "clean_chars": len(norm_text),
        "clean_bytes": len(norm_text.encode("utf-8")),
        "bengali_ratio": ratio,
        "passed_filter": passed_filter,
        "norm_stats": norm_stats,
    }

    return norm_text, passed_filter, stats
