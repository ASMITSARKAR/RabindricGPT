"""
src/rgpt/tokenizers/base.py : Abstract Base Class defining the unified tokenizer contract across all variants.
Backlog/Phase : I3 / Phase E      Docs : 15_BUILD_DIRECTIVE.md Phase E; 04_TOKENIZER_PLAN.md §1
Contract      : encode(text: str) -> list[int]; decode(ids: list[int]) -> str
Invariants    : decode(encode(x)) == x for all lossless variants; vocab_size is fixed; specials consistent
Failure modes : Token ID out of vocab range; invalid byte sequences in byte decoders
Status        : TESTED
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union


class TokenizerError(Exception):
    """Base exception for tokenizer failures."""
    pass


class BaseTokenizer(ABC):
    """Abstract Base Class for all RabindricGPT tokenizers."""

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    UNK_ID: int = 3

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to a text string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded string.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Total vocabulary size including specials and fallbacks."""
        raise NotImplementedError

    @abstractmethod
    def to_files(self, directory: Union[str, Path]) -> None:
        """Save vocabulary and metadata files to directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_files(cls, directory: Union[str, Path]) -> "BaseTokenizer":
        """Instantiate tokenizer from saved directory."""
        raise NotImplementedError

    @abstractmethod
    def content_hash(self) -> str:
        """Return deterministic SHA256 hash of the tokenizer vocabulary and config."""
        raise NotImplementedError
