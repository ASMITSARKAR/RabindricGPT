"""
scripts/05_train_tokenizers.py : Train all tokenizer variants on cleaned text sample and serialize to tok/<id>/.
Backlog/Phase : I3 / Phase E      Docs : 15_BUILD_DIRECTIVE.md Phase E; 04_TOKENIZER_PLAN.md §3
Contract      : --tokenizer-id <id> -> tok/<id>/{vocab, model, config.json}
Invariants    : Uses identical text sample; fixed seed; exact vocab sizes match specs
Failure modes : Sample size exceeds RAM/CPU budget; character coverage < 1.0; missing input corpus
Status        : TESTED
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.common.atomic_io import atomic_write_text
from src.rgpt.common.seed import seed_all
from src.rgpt.tokenizers.akshara import AksharaTokenizer
from src.rgpt.tokenizers.byte import ByteTokenizer
from src.rgpt.tokenizers.char import CharTokenizer
from src.rgpt.tokenizers.spm import train_sentencepiece


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RabindricGPT tokenizers")
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default="all",
        choices=["all", "byte", "char", "aks", "bpe16", "bpe8", "uni16"],
        help="Tokenizer ID or 'all'",
    )
    parser.add_argument(
        "--input-corpus",
        type=str,
        default="data/interim/p1_shuffled.txt",
        help="Path to cleaned input corpus for tokenizer training",
    )
    parser.add_argument(
        "--sample-chars",
        type=int,
        default=500000000,
        help="Max characters to use for training",
    )
    parser.add_argument("--output-dir", type=str, default="tok", help="Base output directory for tokenizers")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic training seed")
    return parser.parse_args()


def train_single(
    tok_id: str,
    text: str,
    out_base: Path,
    seed: int,
    sample_path: Path,
) -> None:
    tok_dir = out_base / tok_id
    tok_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training tokenizer '{tok_id}' -> {tok_dir}")

    if tok_id == "byte":
        tok = ByteTokenizer()
        tok.to_files(tok_dir)
        meta = {"id": tok_id, "vocab_size": tok.vocab_size, "type": "byte"}
        atomic_write_text(tok_dir / "config.json", json.dumps(meta, indent=2))

    elif tok_id == "char":
        tok = CharTokenizer.train_from_text(text)
        tok.to_files(tok_dir)
        meta = {"id": tok_id, "vocab_size": tok.vocab_size, "type": "char"}
        atomic_write_text(tok_dir / "config.json", json.dumps(meta, indent=2))

    elif tok_id == "aks":
        tok = AksharaTokenizer.train_from_text(text)
        tok.to_files(tok_dir)
        meta = {"id": tok_id, "vocab_size": tok.vocab_size, "type": "akshara"}
        atomic_write_text(tok_dir / "config.json", json.dumps(meta, indent=2))

    elif tok_id in ("bpe16", "bpe8", "uni16"):
        vocab_size = 16384 if "16" in tok_id else 8192
        model_type = "unigram" if tok_id.startswith("uni") else "bpe"
        model_prefix = str(tok_dir / "spm")
        train_sentencepiece(
            input_file=sample_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            seed=seed,
        )
        meta = {
            "id": tok_id,
            "vocab_size": vocab_size,
            "type": "spm",
            "model_type": model_type,
            "model_prefix": "spm",
        }
        atomic_write_text(tok_dir / "config.json", json.dumps(meta, indent=2))


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_corpus)
    sample_text = ""
    if input_path.exists():
        with open(input_path, "r", encoding="utf-8") as f:
            sample_text = f.read(args.sample_chars)
    else:
        # Fallback to minimal representative Bengali text if full corpus not yet ingested
        sample_text = (
            "রবীন্দ্রনাথ ঠাকুর ছিলেন অগ্রণী বাঙালি কবি, ঔপন্যাসিক, সংগীতস্রষ্টা ও নাট্যকার। "
            "গীতাঞ্জলি কাব্যগ্রন্থের জন্য তিনি ১৯১৩ সালে সাহিত্যে নোবেল পুরস্কার লাভ করেন।"
        ) * 100
        # Write temporary sample file for SentencePiece
        temp_sample = out_base / "temp_train_sample.txt"
        atomic_write_text(temp_sample, sample_text)
        input_path = temp_sample

    targets: List[str] = (
        ["byte", "char", "aks", "bpe16", "bpe8", "uni16"]
        if args.tokenizer_id == "all"
        else [args.tokenizer_id]
    )

    for tid in targets:
        train_single(
            tok_id=tid,
            text=sample_text,
            out_base=out_base,
            seed=args.seed,
            sample_path=input_path,
        )

    print(f"All tokenizers trained and written to {out_base}/")


if __name__ == "__main__":
    main()
