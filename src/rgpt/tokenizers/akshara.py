"""
src/rgpt/tokenizers/akshara.py : Akshara-based tokenizer with vocab exactly 8192, using char and byte fallbacks.
Backlog/Phase : I3 / Phase E      Docs : 15_BUILD_DIRECTIVE.md §2; 04_TOKENIZER_PLAN.md §1, §2
Contract      : encode(text: str) -> list[int]; decode(ids: list[int]) -> str
Invariants    : Vocab size exactly 8192; encode by longest match (akshara -> char -> byte); lossless
Failure modes : Token ID out of range; invalid byte fallback sequences
Status        : TESTED
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.rgpt.common.atomic_io import atomic_write_text
from src.rgpt.common.hashing import hash_text
from src.rgpt.text.akshara import segment
from src.rgpt.tokenizers.base import BaseTokenizer


class AksharaTokenizer(BaseTokenizer):
    """Akshara-level tokenizer with character and byte fallbacks, exactly 8192 vocab."""

    VOCAB_SIZE: int = 8192
    NUM_SPECIALS: int = 4  # PAD=0, BOS=1, EOS=2, UNK=3
    BYTE_FALLBACK_OFFSET: int = 4  # 4..259 (256 byte tokens)
    PIECE_START_ID: int = 260

    def __init__(self, vocab: Optional[Dict[str, int]] = None) -> None:
        self.piece_to_id: Dict[str, int] = dict(vocab) if vocab else {}
        self.id_to_piece: Dict[int, str] = {idx: piece for piece, idx in self.piece_to_id.items()}

    @classmethod
    def train_from_text(cls, text: str) -> "AksharaTokenizer":
        """Build akshara vocabulary from corpus, filling up to exactly 8192 slots."""
        # 1. Segment text into aksharas
        units = segment(text)
        counts = Counter(units)

        # Base single characters to guarantee decomposition
        all_chars = set(text)
        vocab: Dict[str, int] = {}
        curr_id = cls.PIECE_START_ID

        # Add single characters first
        for ch in sorted(list(all_chars)):
            if curr_id < cls.VOCAB_SIZE:
                vocab[ch] = curr_id
                curr_id += 1

        # Add most frequent multi-char aksharas
        for piece, _ in counts.most_common():
            if piece not in vocab and curr_id < cls.VOCAB_SIZE:
                vocab[piece] = curr_id
                curr_id += 1

        # Pad any remaining slots with reserved tokens to ensure exactly 8192
        pad_idx = 0
        while curr_id < cls.VOCAB_SIZE:
            placeholder = f"<reserved_aks_{pad_idx}>"
            vocab[placeholder] = curr_id
            curr_id += 1
            pad_idx += 1

        return cls(vocab=vocab)

    def encode(self, text: str) -> List[int]:
        """Encode text using akshara segmentation with fallback to characters and bytes."""
        units = segment(text)
        ids: List[int] = []

        for unit in units:
            if unit in self.piece_to_id:
                ids.append(self.piece_to_id[unit])
            else:
                # Fallback to single characters
                for ch in unit:
                    if ch in self.piece_to_id:
                        ids.append(self.piece_to_id[ch])
                    else:
                        # Fallback to bytes
                        for b in ch.encode("utf-8"):
                            ids.append(self.BYTE_FALLBACK_OFFSET + b)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to string, resolving byte fallbacks."""
        pieces: List[str] = []
        pending_bytes: List[int] = []

        def flush_bytes():
            if pending_bytes:
                pieces.append(bytes(pending_bytes).decode("utf-8", errors="replace"))
                pending_bytes.clear()

        for tid in ids:
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            if self.BYTE_FALLBACK_OFFSET <= tid < self.PIECE_START_ID:
                pending_bytes.append(tid - self.BYTE_FALLBACK_OFFSET)
            elif tid in self.id_to_piece:
                flush_bytes()
                piece = self.id_to_piece[tid]
                if not piece.startswith("<reserved_aks_"):
                    pieces.append(piece)
            elif tid == self.UNK_ID:
                flush_bytes()
                pieces.append("\uFFFD")
            else:
                flush_bytes()

        flush_bytes()
        return "".join(pieces)

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def to_files(self, directory: Union[str, Path]) -> None:
        """Save akshara vocabulary to JSON file."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "id": "aks",
            "vocab_size": self.VOCAB_SIZE,
            "piece_to_id": self.piece_to_id,
        }
        atomic_write_text(out_dir / "vocab.json", json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def from_files(cls, directory: Union[str, Path]) -> "AksharaTokenizer":
        """Load akshara tokenizer from directory."""
        v_path = Path(directory) / "vocab.json"
        data = json.loads(v_path.read_text(encoding="utf-8"))
        return cls(vocab=data.get("piece_to_id", {}))

    def content_hash(self) -> str:
        """Deterministic hash of akshara vocabulary."""
        return hash_text(json.dumps(self.piece_to_id, sort_keys=True, ensure_ascii=False))
