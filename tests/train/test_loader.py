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
    """loss_mask must be False for all PAD tokens and True for content tokens when static padding is used."""
    # Chunk with fewer tokens than s_max=64
    text = "ছোট বাক্য।"
    chunks = chunk_document(text, doc_id="short", source="wikisource", split="train")
    tok = ByteTokenizer()
    writer = ShardWriter(output_dir=tmp_path / "pad_shards", tokenizer=tok, source="wikisource", split="train")
    writer.write_chunks(chunks)

    reader = ShardReader(tmp_path / "pad_shards" / "index.json")
    loader = DeterministicDataLoader(
        reader=reader, batch_size=1, s_max=64, seed=42, pad_id=0, pad_to_batch_max=False
    )

    in_ids, targets, loss_mask = loader.get_batch(step=0)

    assert in_ids.shape == (1, 64)
    assert targets.shape == (1, 64)
    assert loss_mask.shape == (1, 64)

    # Content length
    raw_tokens = tok.encode(text)
    content_len = len(raw_tokens) - 1

    assert loss_mask[0, :content_len].all().item() is True
    assert (~loss_mask[0, content_len:]).all().item() is True


def test_loader_dynamic_batch_max_padding(tmp_path: Path) -> None:
    """Dynamic padding must pad only to max sequence length in the batch."""
    text = "ছোট বাক্য।"
    chunks = chunk_document(text, doc_id="short2", source="wikisource", split="train")
    tok = ByteTokenizer()
    writer = ShardWriter(output_dir=tmp_path / "dyn_shards", tokenizer=tok, source="wikisource", split="train")
    writer.write_chunks(chunks)

    reader = ShardReader(tmp_path / "dyn_shards" / "index.json")
    loader = DeterministicDataLoader(
        reader=reader, batch_size=1, s_max=64, seed=42, pad_id=0, pad_to_batch_max=True
    )

    in_ids, targets, loss_mask = loader.get_batch(step=0)
    raw_tokens = tok.encode(text)
    content_len = len(raw_tokens) - 1

    assert in_ids.shape == (1, content_len)
    assert targets.shape == (1, content_len)
    assert loss_mask.shape == (1, content_len)
    assert loss_mask.all().item() is True


def test_loss_equality_dynamic_vs_static_padding() -> None:
    """Item 3: Loss on 5 batches must be numerically identical between dynamic and static padding."""
    import torch.nn.functional as F

    from src.rgpt.model.config import ModelConfig
    from src.rgpt.model.gpt import RabindricGPT

    shard_path = Path("shards/wikisource/bpe16/train")
    if not shard_path.exists():
        pytest.skip(f"Shard directory missing at {shard_path}; mount Drive shards first.")

    with ShardReader(shard_path) as reader:
        cfg = ModelConfig(
            size_id="10m",
            n_layers=2,
            d_model=128,
            head_dim=32,
            n_heads=4,
            n_kv_heads=4,
            hidden_dim=384,
            vocab_size=16384,
            max_seq_len=3072,
        )
        model = RabindricGPT(cfg)
        model.eval()

        loader_static = DeterministicDataLoader(
            reader=reader, batch_size=4, s_max=3072, pad_to_batch_max=False
        )
        loader_dyn = DeterministicDataLoader(
            reader=reader, batch_size=4, s_max=3072, pad_to_batch_max=True
        )

        for step in range(5):
            in_stat, tgt_stat, mask_stat = loader_static.get_batch(step)
            in_dyn, tgt_dyn, mask_dyn = loader_dyn.get_batch(step)

            batch_max = in_dyn.shape[1]
            assert in_stat[:, :batch_max].equal(in_dyn)

            with torch.no_grad():
                logits_stat = model(in_stat)[:, :batch_max, :]
                logits_dyn = model(in_dyn)

                # Loss on static padded batch (restricted to valid non-padded tokens)
                raw_stat = F.cross_entropy(
                    logits_stat.reshape(-1, cfg.vocab_size),
                    tgt_stat[:, :batch_max].reshape(-1),
                    reduction="none",
                )
                loss_stat = (raw_stat * mask_dyn.reshape(-1).float()).sum() / mask_dyn.sum().float()

                # Loss on dynamic padded batch
                raw_dyn = F.cross_entropy(
                    logits_dyn.reshape(-1, cfg.vocab_size),
                    tgt_dyn.reshape(-1),
                    reduction="none",
                )
                loss_dyn = (raw_dyn * mask_dyn.reshape(-1).float()).sum() / mask_dyn.sum().float()

                diff = torch.abs(loss_stat - loss_dyn).item()
                assert diff < 1e-5, f"Loss mismatch at step {step}: diff={diff}"

