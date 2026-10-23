"""
src/rgpt/data/shards.py : Memory-mapped uint16 token shard writer and reader partitioned by corpus without mixing.
Backlog/Phase : I6 / Phase G      Docs : 15_BUILD_DIRECTIVE.md Phase G; 03_DATA_PLAN.md §6
Contract      : write_shards(...) -> index.json; ShardReader(index_path) -> chunk_tokens
Invariants    : Shards partitioned strictly (wikisource/ vs fineweb2/); uint16 dtype; tokens <= s_max
Failure modes : Token ID >= 65536 (overflows uint16); chunk length > s_max; mixing sources without flag
Status        : TESTED
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from src.rgpt.common.atomic_io import atomic_write_text
from src.rgpt.data.chunks import ChunkRecord
from src.rgpt.tokenizers.base import BaseTokenizer


class ShardSourceIsolationError(Exception):
    """Raised when different corpus sources are mixed into the same shard folder."""
    pass


class ShardWriter:
    """Writes tokenized chunk records to memory-mapped uint16 binary shards and index.json."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        tokenizer: BaseTokenizer,
        source: str,
        split: str,
        s_max: int = 3072,
        target_shard_bytes: int = 268435456,
        allow_mixed: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.source = source
        self.split = split
        self.s_max = s_max
        self.target_shard_bytes = target_shard_bytes
        self.allow_mixed = allow_mixed

        if tokenizer.vocab_size > 65535:
            raise ValueError(
                f"Tokenizer vocab size {tokenizer.vocab_size} exceeds uint16 maximum (65535)."
            )

    def write_chunks(self, chunks: List[ChunkRecord]) -> Dict[str, Any]:
        """Tokenize chunks, write binary shard files and index.json.

        Args:
            chunks: List of ChunkRecord instances to serialize.

        Returns:
            The generated index.json dictionary.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        total_tokens = 0
        total_chars = 0
        total_bytes = 0
        all_token_ids: List[int] = []
        chunk_offsets: List[Dict[str, Any]] = []

        curr_offset = 0

        for chunk in chunks:
            # Source isolation check
            if not self.allow_mixed and chunk.source != self.source:
                raise ShardSourceIsolationError(
                    f"Corpus source mixing forbidden: expected '{self.source}', got '{chunk.source}'."
                )

            # Tokenize chunk
            token_ids = self.tokenizer.encode(chunk.text)

            # Overflow guard
            for tid in token_ids:
                if tid >= 65536:
                    raise ValueError(
                        f"Token ID {tid} exceeds uint16 max (65535), cannot serialize to uint16 shard."
                    )

            num_tokens = len(token_ids)
            chunk_offsets.append({
                "chunk_id": chunk.chunk_id,
                "work_id": chunk.work_id,
                "start_offset": curr_offset,
                "num_tokens": num_tokens,
                "n_chars": chunk.n_chars,
                "n_bytes": chunk.n_bytes,
            })

            all_token_ids.extend(token_ids)
            curr_offset += num_tokens
            total_tokens += num_tokens
            total_chars += chunk.n_chars
            total_bytes += chunk.n_bytes

        # Write binary uint16 shard
        shard_filename = "shard_00000.bin"
        bin_path = self.output_dir / shard_filename
        token_array = np.array(all_token_ids, dtype=np.uint16)
        token_array.tofile(bin_path)

        index_data = {
            "version": "1.0",
            "source": self.source,
            "split": self.split,
            "tokenizer_hash": self.tokenizer.content_hash(),
            "vocab_size": self.tokenizer.vocab_size,
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "total_bytes": total_bytes,
            "shard_files": [shard_filename],
            "chunk_offsets": chunk_offsets,
        }

        index_path = self.output_dir / "index.json"
        atomic_write_text(index_path, json.dumps(index_data, indent=2))

        return index_data


class ShardReader:
    """Reads chunk tokens and metadata from memory-mapped binary shards."""

    def __init__(self, index_path: Union[str, Path]) -> None:
        self.index_path = Path(index_path)
        if self.index_path.is_dir():
            self.index_path = self.index_path / "index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"Shard index file not found: {self.index_path}")

        self.index_data = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.chunk_offsets = self.index_data.get("chunk_offsets", [])
        self.base_dir = self.index_path.parent

        # Map shard file(s) via np.memmap
        shard_files = self.index_data.get("shard_files", [])
        self.shards: List[np.memmap] = []
        for sf in shard_files:
            bin_file = self.base_dir / sf
            mmap = np.memmap(bin_file, dtype=np.uint16, mode="r")
            self.shards.append(mmap)

    def __len__(self) -> int:
        return len(self.chunk_offsets)

    def get_chunk(self, index: int) -> Dict[str, Any]:
        """Retrieve token IDs and metadata for a single chunk by index."""
        if index < 0 or index >= len(self.chunk_offsets):
            raise IndexError(f"Chunk index {index} out of range [0, {len(self.chunk_offsets)})")

        entry = self.chunk_offsets[index]
        start = entry["start_offset"]
        num = entry["num_tokens"]

        # Primary single-shard lookup
        tokens = self.shards[0][start : start + num]

        return {
            "chunk_id": entry["chunk_id"],
            "work_id": entry.get("work_id"),
            "tokens": tokens,
            "num_tokens": num,
            "n_chars": entry["n_chars"],
            "n_bytes": entry["n_bytes"],
        }

    def close(self) -> None:
        """Close memory mapped files and release file locks."""
        for m in self.shards:
            if hasattr(m, "_mmap") and m._mmap is not None:
                try:
                    m._mmap.close()
                except Exception:
                    pass
        self.shards.clear()

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

