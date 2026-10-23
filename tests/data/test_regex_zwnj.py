"""
tests/data/test_regex_zwnj.py : Unit test proving that stripping ZWJ (\\u200D) and ZWNJ (\\u200C)
before Bengali regex tokenization prevents improper splitting of ya-phala (র্য-type) words.
"""

import re
from src.rgpt.data.leakage import tokenize_bengali_words


def test_zwnj_zwj_word_splitting_prevented():
    """Prove regex properly keeps words with internal ZWJ/ZWNJ intact."""
    # Word with ZWJ in ya-phala: 'সূর্য' + ZWJ + '্য'
    text_with_zwj = "সূর্য\u200d্য এবং সৌন্দর্য\u200cপূর্ণ পরিবেশ।"

    # Native regex without stripping splits at ZWJ
    naive_tokens = re.findall(r"[\u0980-\u09FF]+", text_with_zwj)
    assert "্য" in naive_tokens, "Naive regex incorrectly split ZWJ conjunct into separate token!"
    assert "সূর্য্য" not in naive_tokens

    # Robust tokenizer with stripping retains the full words
    tokens = tokenize_bengali_words(text_with_zwj)
    assert "সূর্য্য" in tokens
    assert "সৌন্দর্যপূর্ণ" in tokens
    assert "্য" not in tokens
    assert tokens == ["সূর্য্য", "এবং", "সৌন্দর্যপূর্ণ", "পরিবেশ"]
