"""
src/rgpt/text/akshara.py : Orthographic syllable (akshara) segmentation engine for Bengali script.
Backlog/Phase : I2 / Phase D      Docs : 15_BUILD_DIRECTIVE.md Phase D; 04_TOKENIZER_PLAN.md §2
Contract      : text: str -> list[str]
Invariants    : "".join(segment(x)) == x (lossless concatenation); no unit starts with dependent mark
Failure modes : Malformed orphan marks handled gracefully as single units; unbounded recursion prevented
Status        : TESTED
"""

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
)

DEPENDENT_MARKS = VOWEL_SIGNS | MODIFIERS | {VIRAMA, NUKTA}
START_CHARS = CONSONANTS | INDEPENDENT_VOWELS | {KHANDA_TA, AVAGRAHA}


def segment(text: str) -> List[str]:
    """Segment Bengali text into linguistic akshara units using longest-match grammar.

    Grammar (04_TOKENIZER_PLAN.md §2.2):
        C        := consonant [NUKTA]
        CLUSTER  := C ( VIRAMA [ZWJ|ZWNJ]? C )*
        AKSHARA  := ( CLUSTER | INDEP_VOWEL | KHANDA_TA | AVAGRAHA )
                    [ VOWEL_SIGN ]* [ MODIFIERS ]*
                    [ VIRAMA ]?
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

                # Conjunct chain: (VIRAMA [ZWJ|ZWNJ]? C [NUKTA])*
                while i < n and text[i] == VIRAMA:
                    lookahead = i + 1
                    # Check optional ZWJ/ZWNJ after virama
                    if lookahead < n and text[lookahead] in (ZWJ, ZWNJ):
                        lookahead += 1

                    if lookahead < n and text[lookahead] in CONSONANTS:
                        # Valid conjunct extension
                        i = lookahead + 1
                        if i < n and text[i] == NUKTA:
                            i += 1
                    else:
                        # Trailing hasanta / word-final virama
                        break
            else:
                # INDEPENDENT_VOWEL, KHANDA_TA, or AVAGRAHA
                i += 1

            # Attach vowel signs
            while i < n and text[i] in VOWEL_SIGNS:
                i += 1

            # Attach modifiers (chandrabindu, anusvara, visarga)
            while i < n and text[i] in MODIFIERS:
                i += 1

            # Optional word-final virama (hasanta)
            if i < n and text[i] == VIRAMA:
                i += 1
                # Optional trailing ZWJ/ZWNJ
                if i < n and text[i] in (ZWJ, ZWNJ):
                    i += 1

            units.append(text[start_idx:i])
            continue

        # Rule 4: Everything else is one unit per code point
        units.append(ch)
        i += 1

    return units
