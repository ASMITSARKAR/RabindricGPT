"""
tests/tokenizers/test_char.py : Tests for Unicode character tokenizer, <=512 bound, and byte fallback.
"""

import pytest

from src.rgpt.tokenizers.base import TokenizerError
from src.rgpt.tokenizers.char import CharTokenizer


def test_char_tokenizer_vocab_bound() -> None:
    """Character tokenizer vocab must never exceed 512."""
    corpus = "গীতাঞ্জলি কাব্যগ্রন্থের জন্য রবীন্দ্রনাথ ঠাকুর নোবেল পুরস্কার পান।"
    tok = CharTokenizer.train_from_text(corpus, max_chars=100)
    assert tok.vocab_size <= 512


def test_char_tokenizer_round_trip() -> None:
    """Character tokenizer must round-trip common and rare Bengali characters via byte fallback."""
    corpus = "বাংলা বর্ণমালা: অ আ ক খ গ ঘ"
    tok = CharTokenizer.train_from_text(corpus)

    test_str = "বাংলা বর্ণমালা ছাড়াও rare symbol: 🌟 and English!"
    ids = tok.encode(test_str)
    decoded = tok.decode(ids)
    assert decoded == test_str


def test_char_tokenizer_overflow_escalation() -> None:
    """Exceeding 512 tokens must raise TokenizerError per T2 policy."""
    # Attempt to initialize with > 252 characters (which would exceed 512 with 260 specials/bytes)
    overflow_vocab = {f"char_{i}": 260 + i for i in range(300)}
    with pytest.raises(TokenizerError, match="exceeds 512 limit"):
        CharTokenizer(vocab=overflow_vocab)
