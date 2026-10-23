"""
tests/conftest.py : PyTest shared fixtures and configuration for RabindricGPT.
"""

import sys
from pathlib import Path

import pytest

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def mini_corpus_dir() -> Path:
    """Path to tiny synthetic test corpus."""
    return Path(__file__).parent / "data" / "mini"
