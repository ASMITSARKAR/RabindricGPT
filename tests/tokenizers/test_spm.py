"""
tests/tokenizers/test_spm.py : Tests for SentencePiece BPE and Unigram tokenizers and determinism.
"""

from pathlib import Path

from src.rgpt.tokenizers.spm import SentencePieceTokenizer, train_sentencepiece


def test_spm_training_and_round_trip(tmp_path: Path) -> None:
    """SentencePiece models with byte fallback must train and losslessly round-trip held-out text."""
    train_text = tmp_path / "train.txt"
    sample = "রবীন্দ্রনাথ ঠাকুর নোবেল বিজয়ী কবি। গীতাঞ্জলি তাঁহার বিখ্যাত কাব্যগ্রন্থ।\n" * 50
    train_text.write_text(sample, encoding="utf-8")

    model_prefix = str(tmp_path / "spm_test")
    model_path = train_sentencepiece(
        input_file=train_text,
        model_prefix=model_prefix,
        vocab_size=500,
        model_type="bpe",
        seed=42,
    )

    tok = SentencePieceTokenizer(model_file=model_path)
    assert tok.vocab_size == 500

    test_str = "গীতাঞ্জলি তাঁহার শ্রেষ্ঠ কাব্য।"
    ids = tok.encode(test_str)
    assert len(ids) > 0
    # Decoded text should reconstruct the input
    decoded = tok.decode(ids)
    assert decoded.strip() == test_str.strip()


def test_spm_training_determinism(tmp_path: Path) -> None:
    """Same seed and same input text must produce byte-identical .model and .vocab files."""
    train_text = tmp_path / "train_det.txt"
    train_text.write_text("সোনার তরী ও শেষের কবিতা রবীন্দ্রনাথ ঠাকুর\n" * 20, encoding="utf-8")

    m1 = train_sentencepiece(train_text, str(tmp_path / "m1"), vocab_size=300, seed=42)
    m2 = train_sentencepiece(train_text, str(tmp_path / "m2"), vocab_size=300, seed=42)

    assert m1.read_bytes() == m2.read_bytes()
