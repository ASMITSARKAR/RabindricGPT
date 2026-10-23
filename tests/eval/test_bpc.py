"""
tests/eval/test_bpc.py : Tests for Bits-Per-Character (BPC) and Bits-Per-Byte (BPB) evaluation.
"""

import math
import tempfile

import torch
import torch.nn as nn

from src.rgpt.data.chunks import chunk_document
from src.rgpt.data.shards import ShardReader, ShardWriter
from src.rgpt.eval.bpc import evaluate_bpc
from src.rgpt.tokenizers.char import CharTokenizer


class UniformDummyModel(nn.Module):
    """Dummy model outputting uniform logits across vocabulary."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape
        # Return constant logits (so softmax is 1/V everywhere)
        return torch.zeros((b, s, self.vocab_size), dtype=torch.float32)


def test_bpc_uniform_model_theoretical_limit() -> None:
    """A uniform random model must produce BPC equal to log2(V) / (chars_per_token)."""
    text = "গীতাঞ্জলি কাব্যগ্রন্থের জন্য রবীন্দ্রনাথ ঠাকুর নোবেল পান।"
    tok = CharTokenizer.train_from_text(text)
    vocab_size = tok.vocab_size
    model = UniformDummyModel(vocab_size)

    with tempfile.TemporaryDirectory() as tmp_dir:
        chunks = chunk_document(
            text=text,
            doc_id="test_doc",
            source="wikisource",
            split="val",
            chunk_size_chars=1024,
        )
        writer = ShardWriter(
            output_dir=tmp_dir,
            tokenizer=tok,
            source="wikisource",
            split="val",
        )
        writer.write_chunks(chunks)

        with ShardReader(tmp_dir) as reader:
            res = evaluate_bpc(model, reader, device="cpu")

            chunk = reader.get_chunk(0)
            n_tokens = len(chunk["tokens"])
            n_chars = chunk["n_chars"]

            # Predicted tokens: n_tokens - 1 (since BOS unscored)
            predicted_tokens = n_tokens - 1
            expected_nats = predicted_tokens * math.log(vocab_size)
            expected_bpc = expected_nats / (n_chars * math.log(2.0))

            assert math.isclose(res["bpc"], expected_bpc, rel_tol=1e-4)


def test_bos_unscored_in_bpc() -> None:
    """BOS token NLL must be strictly excluded from BPC and BPB sums per DR-2."""
    text = "বাংলা ভাষা"
    tok = CharTokenizer.train_from_text(text)
    vocab_size = tok.vocab_size
    model = UniformDummyModel(vocab_size)

    with tempfile.TemporaryDirectory() as tmp_dir:
        chunks = chunk_document(
            text=text,
            doc_id="test_doc2",
            source="wikisource",
            split="val",
            chunk_size_chars=1024,
        )
        writer = ShardWriter(
            output_dir=tmp_dir,
            tokenizer=tok,
            source="wikisource",
            split="val",
        )
        writer.write_chunks(chunks)

        with ShardReader(tmp_dir) as reader:
            res = evaluate_bpc(model, reader, device="cpu")

            chunk = reader.get_chunk(0)
            n_tokens = len(chunk["tokens"])
            expected_nats = (n_tokens - 1) * math.log(vocab_size)
            assert math.isclose(res["total_nll_nats"], expected_nats, rel_tol=1e-4)
