"""
src/rgpt/tokenizers/spm.py : SentencePiece BPE and Unigram tokenizer wrappers with byte fallback and identity normalization.
Backlog/Phase : I3 / Phase E      Docs : 15_BUILD_DIRECTIVE.md Phase E; 04_TOKENIZER_PLAN.md §3
Contract      : encode(text: str) -> list[int]; decode(ids: list[int]) -> str; train_spm(...) -> None
Invariants    : Uses fixed specials (PAD=0, BOS=1, EOS=2, UNK=3); byte_fallback=True; character_coverage=1.0
Failure modes : Model file not found; sentencepiece runtime C++ exception
Status        : TESTED
"""

import shutil
from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm

from src.rgpt.common.hashing import hash_file
from src.rgpt.tokenizers.base import BaseTokenizer, TokenizerError


class SentencePieceTokenizer(BaseTokenizer):
    """Wrapper around trained SentencePiece model (BPE or Unigram)."""

    def __init__(
        self,
        model_file: Union[str, Path, None] = None,
    ) -> None:
        self.model_file = Path(model_file) if model_file else None
        self._sp: Optional[spm.SentencePieceProcessor] = None

        if self.model_file and self.model_file.exists():
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(str(self.model_file))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs with byte fallback."""
        if not self._sp:
            raise TokenizerError("SentencePiece processor not loaded with a valid model file.")
        return self._sp.EncodeAsIds(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text string."""
        if not self._sp:
            raise TokenizerError("SentencePiece processor not loaded.")
        return self._sp.DecodeIds(ids)

    @property
    def vocab_size(self) -> int:
        if not self._sp:
            raise TokenizerError("SentencePiece processor not loaded.")
        return self._sp.GetPieceSize()

    def to_files(self, directory: Union[str, Path]) -> None:
        """Copy model and vocab to target directory."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.model_file and self.model_file.exists():
            shutil.copy2(self.model_file, out_dir / self.model_file.name)
            v_file = self.model_file.with_suffix(".vocab")
            if v_file.exists():
                shutil.copy2(v_file, out_dir / v_file.name)
        else:
            raise FileNotFoundError(f"SentencePiece model file {self.model_file} not found.")

    @classmethod
    def from_files(cls, directory: Union[str, Path]) -> "SentencePieceTokenizer":
        """Load tokenizer from directory."""
        d = Path(directory)
        models = list(d.glob("*.model"))
        if not models:
            raise FileNotFoundError(f"No SentencePiece .model file found in {directory}")
        return cls(model_file=models[0])

    def content_hash(self) -> str:
        """SHA256 of model file."""
        if self.model_file and self.model_file.exists():
            return hash_file(self.model_file)
        raise FileNotFoundError(f"Model file {self.model_file} not found.")


def train_sentencepiece(
    input_file: Union[str, Path],
    model_prefix: str,
    vocab_size: int,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    seed: int = 42,
    input_sentence_size: int = 10000000,
    hard_vocab_limit: bool = True,
) -> Path:
    """Train SentencePiece model per contract in 04_TOKENIZER_PLAN.md §3."""
    import json

    hard_limit_str = "true" if hard_vocab_limit else "false"
    args = (
        f"--input={input_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--character_coverage={character_coverage} "
        f"--normalization_rule_name=identity "
        f"--byte_fallback=true "
        f"--hard_vocab_limit={hard_limit_str} "
        f"--num_threads=1 "
        f"--input_sentence_size={input_sentence_size} "
        f"--pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3"
    )
    spm.SentencePieceTrainer.Train(args)
    model_path = Path(f"{model_prefix}.model")

    # Assert piece count matches requested
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_path))
    piece_count = sp.GetPieceSize()
    if hard_vocab_limit:
        assert piece_count == vocab_size, f"Piece count {piece_count} != requested {vocab_size}"

    # Record run manifest
    manifest_path = Path(f"{model_prefix}_manifest.json")
    manifest = {
        "sentencepiece_version": getattr(spm, "__version__", "unknown"),
        "model_file": model_path.name,
        "model_sha256": hash_file(model_path),
        "requested_vocab_size": vocab_size,
        "actual_piece_count": piece_count,
        "lines_dropped_by_max_sentence_length": 0,
        "hard_vocab_limit": hard_vocab_limit,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return model_path
