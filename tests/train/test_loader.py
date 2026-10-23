"""
tests/train/test_loader.py : Tests for deterministic batch sampler and padding loss-masking.
"""

from pathlib import Path

import torch

from src.rgpt.data.chunks import chunk_document
from src.rgpt.data.shards import ShardReader, ShardWriter
from src.rgpt.tokenizers.byte import ByteTokenizer
from src.rgpt.train.loader import DeterministicDataLoader


def test_loader_deterministic_reproducibility(tmp_path: Path) -> None:
    """Repeated calls with same step must return identical batch indices."""
    text = "বাংলা ভাষা ও সাহিত্য। " * 50
    chunks = chunk_document(text, doc_id="d1", source="wikisource", split="train")
    tok = ByteTokenizer()
    writer = ShardWriter(output_dir=tmp_path / "shards", tokenizer=tok, source="wikisource", split="train")
    writer.write_chunks(chunks)

    reader = ShardReader(tmp_path / "shards" / "index.json")
    loader = DeterministicDataLoader(reader=reader, batch_size=2, s_max=32, seed=42)

    in1, tgt1, mask1 = loader.get_batch(step=10)
    in2, tgt2, mask2 = loader.get_batch(step=10)

    assert torch.equal(in1, in2)
    assert torch.equal(tgt1, tgt2)
    assert torch.equal(mask1, mask2)


def test_loader_loss_mask_covers_padding(tmp_path: Path) -> None:
    """loss_mask must be False for all PAD tokens and True for content tokens."""
    # Chunk with fewer tokens than s_max=64
    text = "ছোট বাক্য।"
    chunks = chunk_document(text, doc_id="short", source="wikisource", split="train")
    tok = ByteTokenizer()
    writer = ShardWriter(output_dir=tmp_path / "pad_shards", tokenizer=tok, source="wikisource", split="train")
    writer.write_chunks(chunks)

    reader = ShardReader(tmp_path / "pad_shards" / "index.json")
    loader = DeterministicDataLoader(reader=reader, batch_size=1, s_max=64, seed=42, pad_id=0)

    in_ids, targets, loss_mask = loader.get_batch(step=0)

    assert in_ids.shape == (1, 64)
    assert targets.shape == (1, 64)
    assert loss_mask.shape == (1, 64)

    # Content length
    raw_tokens = tok.encode(text)
    content_len = len(raw_tokens) - 1

    assert loss_mask[0, :content_len].all().item() is True
    assert (~loss_mask[0, content_len:]).all().item() is True
