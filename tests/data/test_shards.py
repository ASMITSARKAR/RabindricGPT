"""
tests/data/test_shards.py : Tests for memory-mapped uint16 binary shards, equal-bytes invariant, and source isolation.
"""

from pathlib import Path

import pytest

from src.rgpt.data.chunks import chunk_document
from src.rgpt.data.shards import ShardReader, ShardSourceIsolationError, ShardWriter
from src.rgpt.tokenizers.akshara import AksharaTokenizer
from src.rgpt.tokenizers.byte import ByteTokenizer
from src.rgpt.tokenizers.char import CharTokenizer


def test_equal_bytes_invariant_across_tokenizers(tmp_path: Path) -> None:
    """Total character and byte counts in index.json must be identical across all tokenizers for the same split."""
    text = "গীতাঞ্জলি কাব্যগ্রন্থের জন্য রবীন্দ্রনাথ ঠাকুর নোবেল পান। " * 20
    chunks = chunk_document(
        text=text,
        doc_id="test_gora",
        source="wikisource",
        split="val",
        chunk_size_chars=1024,
    )

    tokenizers = {
        "byte": ByteTokenizer(),
        "char": CharTokenizer.train_from_text(text),
        "aks": AksharaTokenizer.train_from_text(text),
    }

    indices = {}
    for name, tok in tokenizers.items():
        out_dir = tmp_path / name
        writer = ShardWriter(output_dir=out_dir, tokenizer=tok, source="wikisource", split="val")
        indices[name] = writer.write_chunks(chunks)

    # Invariant check: chars and bytes MUST be identical across tokenizers
    byte_chars = indices["byte"]["total_chars"]
    byte_bytes = indices["byte"]["total_bytes"]

    for name in ("char", "aks"):
        assert indices[name]["total_chars"] == byte_chars
        assert indices[name]["total_bytes"] == byte_bytes


def test_shard_source_isolation_enforcement(tmp_path: Path) -> None:
    """ShardWriter must refuse to mix wikisource and fineweb2 chunks unless allow_mixed=True."""
    chunks = chunk_document(
        text="ফাইনওয়েব নথি।",
        doc_id="fw_1",
        source="fineweb2",
        split="train",
    )
    tok = ByteTokenizer()
    writer = ShardWriter(
        output_dir=tmp_path / "iso_test",
        tokenizer=tok,
        source="wikisource",
        split="train",
        allow_mixed=False,
    )

    with pytest.raises(ShardSourceIsolationError, match="Corpus source mixing forbidden"):
        writer.write_chunks(chunks)


def test_shard_uint16_overflow_protection(tmp_path: Path) -> None:
    """Token IDs >= 65536 must raise ValueError rather than silently truncating."""
    class FakeOverflowTokenizer(ByteTokenizer):
        def encode(self, text: str):
            return [70000]

    tok = FakeOverflowTokenizer()
    chunks = chunk_document(text="টেক্সট", doc_id="d1", source="wikisource", split="train")
    writer = ShardWriter(output_dir=tmp_path / "overflow", tokenizer=tok, source="wikisource", split="train")

    with pytest.raises(ValueError, match="exceeds uint16 max"):
        writer.write_chunks(chunks)


def test_shard_round_trip(tmp_path: Path) -> None:
    """Reading chunk tokens from binary shard and decoding must reproduce original text."""
    text = "বাংলা ভাষার সৌন্দর্য অপূর্ব।"
    chunks = chunk_document(text=text, doc_id="d1", source="wikisource", split="train")
    tok = ByteTokenizer()
    writer = ShardWriter(output_dir=tmp_path / "rt", tokenizer=tok, source="wikisource", split="train")
    writer.write_chunks(chunks)

    reader = ShardReader(tmp_path / "rt" / "index.json")
    assert len(reader) == 1
    chunk_data = reader.get_chunk(0)
    decoded = tok.decode(chunk_data["tokens"].tolist())
    assert decoded == text
