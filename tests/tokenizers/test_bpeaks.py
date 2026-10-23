"""
tests/tokenizers/test_bpeaks.py : Tests for BPE over Aksharas (stubbed until Gate G4).
"""

import pytest

from src.rgpt.tokenizers.bpeaks import BPEAksharaTokenizer


def test_bpeaks_stub_raises_before_g4() -> None:
    """Instantiating or invoking BPE over Aksharas prior to Gate G4 must raise NotImplementedError."""
    tok = BPEAksharaTokenizer()
    with pytest.raises(NotImplementedError, match="Gate G4"):
        tok.encode("বাংলা")
    with pytest.raises(NotImplementedError, match="Gate G4"):
        tok.decode([1, 2, 3])

