"""
tests/tokenizers/test_base.py : Tests for unified BaseTokenizer contract and interface invariants.
"""

import pytest

from src.rgpt.tokenizers.base import BaseTokenizer


def test_base_tokenizer_abstract_interface() -> None:
    """BaseTokenizer cannot be instantiated directly without implementing abstract methods."""
    with pytest.raises(TypeError):
        BaseTokenizer()  # type: ignore


def test_base_tokenizer_special_ids_consistency() -> None:
    """All tokenizers must share PAD=0, BOS=1, EOS=2, UNK=3."""
    assert BaseTokenizer.PAD_ID == 0
    assert BaseTokenizer.BOS_ID == 1
    assert BaseTokenizer.EOS_ID == 2
    assert BaseTokenizer.UNK_ID == 3
