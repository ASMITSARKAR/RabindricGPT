"""
src/rgpt/text/bengali_unicode.py : Unicode code-point categorization for the Bengali block (U+0980–U+09FF).
Backlog/Phase : I1 / Phase C      Docs : 15_BUILD_DIRECTIVE.md Phase C; 04_TOKENIZER_PLAN.md §2.1
Contract      : char: str -> category: str; is_bengali(char) -> bool
Invariants    : Every class member has the expected Unicode category; no category overlap
Failure modes : Multi-character string passed where single code point expected
Status        : TESTED
"""

import unicodedata
from typing import Set

# 1. Modifiers (Chandrabindu, Anusvara, Visarga)
MODIFIERS: Set[str] = {
    "\u0981",  # Chandrabindu
    "\u0982",  # Anusvara
    "\u0983",  # Visarga
}

# 2. Independent Vowels (U+0985 - U+0994, excluding unassigned 098D, 098E, 0991, 0992)
INDEPENDENT_VOWELS: Set[str] = {
    "\u0985", "\u0986", "\u0987", "\u0988", "\u0989", "\u098a",
    "\u098b", "\u098c", "\u098f", "\u0990", "\u0993", "\u0994",
    "\u09e0", "\u09e1",  # Vocalic RR, LL
}

# 3. Consonants (U+0995 - U+09B9, plus precomposed U+09DC, U+09DD, U+09DF)
CONSONANTS: Set[str] = {
    "\u0995", "\u0996", "\u0997", "\u0998", "\u0999",
    "\u099a", "\u099b", "\u099c", "\u099d", "\u099e",
    "\u099f", "\u09a0", "\u09a1", "\u09a2", "\u09a3",
    "\u09a4", "\u09a5", "\u09a6", "\u09a7", "\u09a8",
    "\u09aa", "\u09ab", "\u09ac", "\u09ad", "\u09ae",
    "\u09af", "\u09b0", "\u09b2", "\u09b6", "\u09b7",
    "\u09b8", "\u09b9",
    "\u09dc", "\u09dd", "\u09df",  # Precomposed RRA, RHA, YYA
    "\u09b1",  # Assamese Ra
    "\u09b5",  # Assamese Va
}

# 4. Signs & Marks
NUKTA: str = "\u09bc"
AVAGRAHA: str = "\u09bd"
VIRAMA: str = "\u09cd"  # Hasanta
KHANDA_TA: str = "\u09ce"

# 5. Dependent Vowel Signs
VOWEL_SIGNS: Set[str] = {
    "\u09be", "\u09bf", "\u09c0", "\u09c1", "\u09c2",
    "\u09c3", "\u09c4", "\u09c7", "\u09c8", "\u09cb",
    "\u09cc", "\u09d7",  # Length mark
    "\u09e2", "\u09e3",  # Vocalic vowel signs
}

# 6. Joiners
ZWJ: str = "\u200d"
ZWNJ: str = "\u200c"

# 7. Digits
DIGITS: Set[str] = {
    "\u09e6", "\u09e7", "\u09e8", "\u09e9", "\u09ea",
    "\u09eb", "\u09ec", "\u09ed", "\u09ee", "\u09ef",
}

# 8. Punctuation
DANDAS: Set[str] = {"\u0964", "\u0965"}  # Danda, Double Danda

# Unassigned code points in U+0980..U+09FF block
BENGALI_UNASSIGNED: Set[str] = {
    chr(cp)
    for cp in range(0x0980, 0x0A00)
    if unicodedata.category(chr(cp)) == "Cn"
}


def is_bengali_char(char: str) -> bool:
    """Return True if character is in Bengali Unicode block or Bengali danda."""
    if len(char) != 1:
        raise ValueError("is_bengali_char requires a single code point")
    cp = ord(char)
    return (0x0980 <= cp <= 0x09FF and char not in BENGALI_UNASSIGNED) or char in DANDAS


def classify_code_point(char: str) -> str:
    """Classify a single Unicode code point into its akshara grammar class."""
    if len(char) != 1:
        raise ValueError("classify_code_point requires a single code point")

    if char in CONSONANTS:
        return "CONSONANT"
    if char in INDEPENDENT_VOWELS:
        return "INDEPENDENT_VOWEL"
    if char in VOWEL_SIGNS:
        return "VOWEL_SIGN"
    if char in MODIFIERS:
        return "MODIFIER"
    if char == VIRAMA:
        return "VIRAMA"
    if char == NUKTA:
        return "NUKTA"
    if char == KHANDA_TA:
        return "KHANDA_TA"
    if char == AVAGRAHA:
        return "AVAGRAHA"
    if char == ZWJ:
        return "ZWJ"
    if char == ZWNJ:
        return "ZWNJ"
    if char in DIGITS:
        return "DIGIT"
    if char in DANDAS:
        return "DANDA"
    if char in BENGALI_UNASSIGNED:
        return "UNASSIGNED"
    if char.isspace():
        return "WHITESPACE"
    return "OTHER"
