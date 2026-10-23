"""
tests/tokenizers/test_intrinsic.py : Tests for intrinsic tokenizer metric computation.
"""

import pytest

from src.rgpt.tokenizers.byte import ByteTokenizer
from src.rgpt.tokenizers.intrinsic import compute_intrinsic_metrics


def test_intrinsic_compression_rate_definition() -> None:
    """Compression rate T must equal UTF-8 bytes divided by tokens."""
    tokenizer = ByteTokenizer()
    text = "বাংলা ভাষা ও সাহিত্য"
    metrics = compute_intrinsic_metrics(tokenizer, text)
    utf8_bytes = len(text.encode("utf-8"))
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0
    assert metrics["bytes_per_token"] == pytest.approx(utf8_bytes / len(tokens))


def test_intrinsic_chars_per_token() -> None:
    """Chars per token must equal text length divided by tokens."""
    tokenizer = ByteTokenizer()
    text = "বাংলা ভাষা"
    metrics = compute_intrinsic_metrics(tokenizer, text)
    tokens = tokenizer.encode(text)
    assert metrics["chars_per_token"] == pytest.approx(len(text) / len(tokens))


def test_intrinsic_metrics_reported() -> None:
    """All intrinsic metrics (fertility, fallback rate, utilization) must be produced."""
    tokenizer = ByteTokenizer()
    text = "রবীন্দ্রনাথ ঠাকুর"
    metrics = compute_intrinsic_metrics(tokenizer, text)
    for key in ["bytes_per_token", "chars_per_token", "fertility", "fallback_rate", "vocab_utilization", "unk_rate"]:
        assert key in metrics
        assert metrics[key] >= 0.0

