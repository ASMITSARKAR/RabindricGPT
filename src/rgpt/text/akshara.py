"""
src/rgpt/text/akshara.py : Orthographic syllable (akshara) segmentation engine for Bengali script.
Backlog/Phase : I2 / Phase D      Docs : 15_BUILD_DIRECTIVE.md Phase D; 04_TOKENIZER_PLAN.md §2
Contract      : text: str -> list[str]
Invariants    : "".join(segment(x)) == x (lossless concatenation); no unit starts with dependent mark
Failure modes : Malformed orphan marks handled gracefully as single units; unbounded recursion prevented
Status        : TESTED
"""

import unicodedata
from typing import List

from src.rgpt.text.bengali_unicode import (
    AVAGRAHA,
    CONSONANTS,
    INDEPENDENT_VOWELS,
    KHANDA_TA,
    MODIFIERS,
    NUKTA,
    VIRAMA,
    VOWEL_SIGNS,
    ZWJ,
    ZWNJ,
    is_bengali_char,
)

DEPENDENT_MARKS = VOWEL_SIGNS | MODIFIERS | {VIRAMA, NUKTA}
START_CHARS = CONSONANTS | INDEPENDENT_VOWELS | {KHANDA_TA, AVAGRAHA}


def segment(text: str) -> List[str]:
    """Segment Bengali text into linguistic akshara units using longest-match grammar.

    Grammar (04_TOKENIZER_PLAN.md §2.2):
        C        := consonant [NUKTA]
        CLUSTER  := C ( [ZWJ|ZWNJ]? VIRAMA [ZWJ|ZWNJ]? C [NUKTA] )*
        V_YA     := INDEP_VOWEL ( [ZWJ|ZWNJ]? VIRAMA [ZWJ|ZWNJ]? YA [NUKTA]? )?
        AKSHARA  := ( CLUSTER | V_YA | KHANDA_TA | AVAGRAHA )
                    [ VOWEL_SIGN ]* [ MODIFIERS ]*
                    [ [ZWJ|ZWNJ]? VIRAMA [ZWJ|ZWNJ]? ]?
        NON_BENG := Extended grapheme cluster (base code point + category M combining marks)
        OTHER    := any other single code point -> its own unit

    Invariants:
        "".join(segment(text)) == text
        No akshara unit starts with a dependent mark unless it is an orphan mark in malformed input.

    Args:
        text: Input string.

    Returns:
        List of segmented substring units.
    """
    if not text:
        return []

    units: List[str] = []
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]

        # Rule 1: Whitespace and newlines are separate units
        if ch.isspace():
            units.append(ch)
            i += 1
            continue

        # Rule 2: Orphan dependent mark becomes its own unit
        if ch in DEPENDENT_MARKS:
            units.append(ch)
            i += 1
            continue

        # Rule 3: Start of an Akshara unit
        if ch in START_CHARS:
            start_idx = i

            if ch in CONSONANTS:
                i += 1
                if i < n and text[i] == NUKTA:
                    i += 1

                # Conjunct chain: ( [ZWJ|ZWNJ]? VIRAMA [ZWJ|ZWNJ]? C [NUKTA] )*
                while i < n:
                    k = i
                    if k < n and text[k] in (ZWJ, ZWNJ):
                        k += 1
                    if k < n and text[k] == VIRAMA:
                        lookahead = k + 1
                        if lookahead < n and text[lookahead] in (ZWJ, ZWNJ):
                            lookahead += 1
                        if lookahead < n and text[lookahead] in CONSONANTS:
                            # Valid conjunct extension
                            i = lookahead + 1
                            if i < n and text[i] == NUKTA:
                                i += 1
                            continue
                        else:
                            # Trailing hasanta / word-final virama attached to this cluster
                            i = k + 1
                            if i < n and text[i] in (ZWJ, ZWNJ):
                                i += 1
                            break
                    else:
                        break

            elif ch in INDEPENDENT_VOWELS:
                i += 1
                # Support vowel + virama + ya-phala as a single unit (e.g. অ্যা, এ্যা, অ্য, এ্য)
                k = i
                if k < n and text[k] in (ZWJ, ZWNJ):
                    k += 1
                if k < n and text[k] == VIRAMA:
                    lookahead = k + 1
                    if lookahead < n and text[lookahead] in (ZWJ, ZWNJ):
                        lookahead += 1
                    if lookahead < n and text[lookahead] in ("\u09af", "\u09df"):  # য, য়
                        i = lookahead + 1
                        if i < n and text[i] == NUKTA:
                            i += 1
            else:
                # KHANDA_TA or AVAGRAHA
                i += 1

            # Attach vowel signs
            while i < n and text[i] in VOWEL_SIGNS:
                i += 1

            # Attach modifiers (chandrabindu, anusvara, visarga)
            while i < n and text[i] in MODIFIERS:
                i += 1

            # Optional word-final virama (hasanta) if not already consumed
            if i < n:
                k = i
                if k < n and text[k] in (ZWJ, ZWNJ):
                    k += 1
                if k < n and text[k] == VIRAMA:
                    i = k + 1
                    if i < n and text[i] in (ZWJ, ZWNJ):
                        i += 1

            units.append(text[start_idx:i])
            continue

        # Rule 4: Non-Bengali / other characters
        # Segment per extended grapheme cluster so non-Bengali base characters and their
        # combining marks (e.g. Arabic fatha/kasra, Latin accents) remain together.
        start_idx = i
        i += 1
        while i < n:
            next_ch = text[i]
            if next_ch.isspace() or next_ch in START_CHARS or next_ch in DEPENDENT_MARKS or is_bengali_char(next_ch):
                break
            if unicodedata.category(next_ch).startswith("M") or next_ch in (ZWJ, ZWNJ):
                i += 1
            else:
                break

        units.append(text[start_idx:i])

    return units
