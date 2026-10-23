"""
tests/eval/test_generation.py : Tests for text generation sampling and Unicode corruption detection.
"""

import pytest
import torch
import torch.nn as nn

from src.rgpt.eval.generation import (
    check_ill_formed_unicode_rate,
    check_in_vocab_word_rate,
    generate_samples,
    is_ill_formed_unicode,
)
from src.rgpt.tokenizers.char import CharTokenizer


def test_ill_formed_unicode_detection() -> None:
    """Checker must detect known invalid UTF-8 byte sequences, surrogates, and replacement chars."""
    good_text = "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।"
    bad_replacement = "সোনার বাংলা \ufffd গান"
    bad_surrogate = "সোনার \ud800 বাংলা"

    assert not is_ill_formed_unicode(good_text)
    assert is_ill_formed_unicode(bad_replacement)
    assert is_ill_formed_unicode(bad_surrogate)

    rate = check_ill_formed_unicode_rate([good_text, bad_replacement, bad_surrogate])
    assert pytest.approx(rate, 0.01) == 2.0 / 3.0


def test_in_vocab_word_rate() -> None:
    """Vocabulary checker must compute fraction of words present in reference vocabulary."""
    text = ["আমার সোনার বাংলা আমি ভালোবাসি"]
    ref_vocab = {"আমার", "বাংলা", "ভালোবাসি"}

    rate = check_in_vocab_word_rate(text, ref_vocab)
    # 3 in-vocab out of 5 words = 0.6
    assert pytest.approx(rate, 0.01) == 3.0 / 5.0


class DummyGenModel(nn.Module):
    """Deterministic next-token model for generation tests."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape
        # Put high logit on a fixed token within vocabulary
        logits = torch.zeros((b, s, self.vocab_size), dtype=torch.float32)
        target_token = min(10, self.vocab_size - 1)
        logits[:, :, target_token] = 50.0  # target token has overwhelming probability
        return logits


def test_generate_samples_with_dummy_model() -> None:
    """generate_samples must produce non-empty completions under nucleus sampling."""
    text = "বাংলা আজ সকালবেলা মেঘের কোলে রোদ"
    tok = CharTokenizer.train_from_text(text)
    model = DummyGenModel(tok.vocab_size)

    prompts = ["বাংলা"]
    samples = generate_samples(
        model=model,
        tokenizer=tok,
        prompts=prompts,
        temperature=0.8,
        top_p=0.95,
        max_new_chars=20,
        seed=42,
        eos_token_id=None,
    )

    assert len(samples) == 1
    assert len(samples[0]) > len("বাংলা")
