"""
tests/tokenizers/test_akshara.py : Tests for akshara tokenizer, exact 8192 vocab, and fallback decomposition.
"""

from src.rgpt.tokenizers.akshara import AksharaTokenizer


def test_akshara_tokenizer_exact_vocab_size() -> None:
    """Akshara tokenizer vocab size must be exactly 8192."""
    corpus = "গীতাঞ্জলি কাব্যগ্রন্থের জন্য রবীন্দ্রনাথ ঠাকুর নোবেল পান।"
    tok = AksharaTokenizer.train_from_text(corpus)
    assert tok.vocab_size == 8192


def test_akshara_tokenizer_round_trip() -> None:
    """Akshara tokenizer must losslessly round-trip held-out Bengali text."""
    corpus = "রবীন্দ্রনাথ ঠাকুরের বিখ্যাত উপন্যাস গোরা এবং নৌকাডুবি।"
    tok = AksharaTokenizer.train_from_text(corpus)

    test_samples = [
        "গোরা এবং নৌকাডুবি",
        "অচেনা শব্দ ও emoji 🌟 সহ টেস্ট!",
        "মিশ্র বাক্য: English and বাংলা মিশ্রণ।",
    ]
    for s in test_samples:
        ids = tok.encode(s)
        decoded = tok.decode(ids)
        assert decoded == s


def test_akshara_tokenizer_fallback_decomposition() -> None:
    """Out-of-vocabulary aksharas must fall back to characters then bytes without UNK."""
    corpus = "ক খ গ ঘ"
    tok = AksharaTokenizer.train_from_text(corpus)

    # 'ক্ষ' was not in training corpus, so it decomposes to 'ক' + '্' + 'ষ' or bytes
    test_str = "ক্ষত্রিয়"
    ids = tok.encode(test_str)
    assert tok.UNK_ID not in ids
    assert tok.decode(ids) == test_str
