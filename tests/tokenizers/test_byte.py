"""
tests/tokenizers/test_byte.py : Tests for byte-level tokenizer vocab size (264) and round-trip losslessness.
"""

import pytest

from src.rgpt.tokenizers.base import TokenizerError
from src.rgpt.tokenizers.byte import ByteTokenizer


def test_byte_tokenizer_exact_vocab_size() -> None:
    """Byte tokenizer vocab size must be exactly 264."""
    tok = ByteTokenizer()
    assert tok.vocab_size == 264


def test_byte_tokenizer_round_trip() -> None:
    """Byte tokenizer must losslessly round-trip arbitrary text, emojis, and control chars."""
    tok = ByteTokenizer()
    samples = [
        "গীতাঞ্জলি - রবীন্দ্রনাথ ঠাকুর",
        "Hello World 123! @#$",
        "Emoji test: 🌟🔥🎉",
        "Whitespace & newlines:\n\n\t  test  ",
    ]
    for s in samples:
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        assert decoded == s


def test_byte_tokenizer_invalid_sequence_handling() -> None:
    """Invalid byte sequences must trigger strict errors as documented."""
    tok = ByteTokenizer()
    # 0xFF is an invalid single byte in UTF-8; in ByteTokenizer offset by 3, so ID = 255 + 3 = 258
    invalid_ids = [258]
    with pytest.raises(TokenizerError, match="failed to decode"):
        tok.decode(invalid_ids, errors="strict")

    # In replace mode, it should not raise
    replaced = tok.decode(invalid_ids, errors="replace")
    assert "\uFFFD" in replaced
