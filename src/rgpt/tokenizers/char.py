"""
src/rgpt/tokenizers/char.py : Unicode character tokenizer capped at <=512 tokens with byte fallback.
Backlog/Phase : I3 / Phase E      Docs : 15_BUILD_DIRECTIVE.md §2; 04_TOKENIZER_PLAN.md §1; T2
Contract      : encode(text: str) -> list[int]; decode(ids: list[int]) -> str
Invariants    : Total vocab size <= 512; byte fallback guarantees losslessness; raises if chars + 256 + specials > 512
Failure modes : Vocab overflow beyond 512 (escalates per T2); unrecognized token ID
Status        : TESTED
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.rgpt.common.atomic_io import atomic_write_text
from src.rgpt.common.hashing import hash_text
from src.rgpt.tokenizers.base import BaseTokenizer, TokenizerError


class CharTokenizer(BaseTokenizer):
    """Unicode character tokenizer with byte fallback."""

    MAX_VOCAB_SIZE: int = 512
    NUM_SPECIALS: int = 4  # PAD=0, BOS=1, EOS=2, UNK=3
    BYTE_FALLBACK_OFFSET: int = 4  # 4..259 (256 byte tokens)
    CHAR_START_ID: int = 260  # 260..511

    def __init__(self, vocab: Optional[Dict[str, int]] = None) -> None:
        # vocab maps char -> id
        self.char_to_id: Dict[str, int] = dict(vocab) if vocab else {}
        self.id_to_char: Dict[int, str] = {idx: ch for ch, idx in self.char_to_id.items()}

        total_v = self.CHAR_START_ID + len(self.char_to_id)
        if total_v > self.MAX_VOCAB_SIZE:
            raise TokenizerError(
                f"CharTokenizer vocabulary exceeds 512 limit: {total_v} tokens. "
                "Per Decision T2, character vocabulary must be capped at <=512."
            )

    @classmethod
    def train_from_text(
        cls, text: str, max_chars: int = 252  # 512 - 260 = 252 available char slots
    ) -> "CharTokenizer":
        """Build character vocabulary from frequent characters in text."""
        from collections import Counter
        counts = Counter(text)
        # Select top characters
        most_common = [char for char, _ in counts.most_common(max_chars)]
        vocab = {char: cls.CHAR_START_ID + idx for idx, char in enumerate(most_common)}
        return cls(vocab=vocab)

    def encode(self, text: str) -> List[int]:
        """Encode text to character token IDs, falling back to byte tokens for rare chars."""
        ids: List[int] = []
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            else:
                # Byte fallback
                for b in ch.encode("utf-8"):
                    ids.append(self.BYTE_FALLBACK_OFFSET + b)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to string, resolving byte fallbacks."""
        chars: List[str] = []
        pending_bytes: List[int] = []

        def flush_bytes():
            if pending_bytes:
                chars.append(bytes(pending_bytes).decode("utf-8", errors="replace"))
                pending_bytes.clear()

        for tid in ids:
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            if self.BYTE_FALLBACK_OFFSET <= tid < self.CHAR_START_ID:
                pending_bytes.append(tid - self.BYTE_FALLBACK_OFFSET)
            elif tid in self.id_to_char:
                flush_bytes()
                chars.append(self.id_to_char[tid])
            elif tid == self.UNK_ID:
                flush_bytes()
                chars.append("\uFFFD")
            else:
                flush_bytes()

        flush_bytes()
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return self.CHAR_START_ID + len(self.char_to_id)

    def to_files(self, directory: Union[str, Path]) -> None:
        """Save character vocabulary to JSON file."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": "char",
            "vocab_size": self.vocab_size,
            "char_to_id": self.char_to_id,
        }
        atomic_write_text(out_dir / "vocab.json", json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def from_files(cls, directory: Union[str, Path]) -> "CharTokenizer":
        """Instantiate character tokenizer from directory."""
        v_path = Path(directory) / "vocab.json"
        data = json.loads(v_path.read_text(encoding="utf-8"))
        return cls(vocab=data.get("char_to_id", {}))

    def content_hash(self) -> str:
        """Deterministic hash of character table."""
        return hash_text(json.dumps(self.char_to_id, sort_keys=True, ensure_ascii=False))
