"""
src/rgpt/tokenizers/byte.py : Pure UTF-8 byte-level tokenizer with padding to vocab size 264.
Backlog/Phase : I3 / Phase E      Docs : 15_BUILD_DIRECTIVE.md §2; 04_TOKENIZER_PLAN.md §1
Contract      : encode(text: str) -> list[int]; decode(ids: list[int]) -> str
Invariants    : Vocab size is 264; byte values 0..255 mapped to IDs 3..258; PAD=0, BOS=1, EOS=2
Failure modes : Decoding invalid UTF-8 byte sequences raises explicit TokenizerError
Status        : TESTED
"""

import json
from pathlib import Path
from typing import List, Union

from src.rgpt.common.atomic_io import atomic_write_text
from src.rgpt.common.hashing import hash_text
from src.rgpt.tokenizers.base import BaseTokenizer, TokenizerError


class ByteTokenizer(BaseTokenizer):
    """Byte-level tokenizer representing raw UTF-8 bytes directly."""

    VOCAB_SIZE: int = 264
    BYTE_OFFSET: int = 3  # PAD=0, BOS=1, EOS=2; Bytes 0..255 -> IDs 3..258

    def __init__(self) -> None:
        pass

    def encode(self, text: str) -> List[int]:
        """Encode text to byte token IDs."""
        byte_data = text.encode("utf-8")
        return [b + self.BYTE_OFFSET for b in byte_data]

    def decode(self, ids: List[int], errors: str = "strict") -> str:
        """Decode token IDs back to UTF-8 text string.

        Args:
            ids: List of integer token IDs.
            errors: 'strict' raises TokenizerError; 'replace' replaces with U+FFFD; 'ignore' skips.

        Returns:
            Decoded text string.

        Raises:
            TokenizerError: If errors='strict' and invalid byte sequence encountered.
        """
        byte_list = []
        for tid in ids:
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            if self.BYTE_OFFSET <= tid <= 258:
                byte_list.append(tid - self.BYTE_OFFSET)
            else:
                # Padding tokens 259..263 or unknown
                continue

        raw_bytes = bytes(byte_list)
        try:
            return raw_bytes.decode("utf-8", errors=errors)
        except UnicodeDecodeError as e:
            raise TokenizerError(f"ByteTokenizer failed to decode byte sequence: {e}") from e

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def to_files(self, directory: Union[str, Path]) -> None:
        """Save byte tokenizer configuration metadata."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": "byte",
            "vocab_size": self.VOCAB_SIZE,
            "byte_offset": self.BYTE_OFFSET,
            "specials": {"pad": self.PAD_ID, "bos": self.BOS_ID, "eos": self.EOS_ID},
        }
        atomic_write_text(out_dir / "tokenizer.json", json.dumps(meta, indent=2))

    @classmethod
    def from_files(cls, directory: Union[str, Path]) -> "ByteTokenizer":
        """Instantiate byte tokenizer from metadata directory."""
        return cls()

    def content_hash(self) -> str:
        """Return deterministic hash for byte tokenizer."""
        return hash_text(f"byte_tokenizer_v1_{self.VOCAB_SIZE}")
