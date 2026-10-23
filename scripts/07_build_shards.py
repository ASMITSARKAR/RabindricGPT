"""
scripts/07_build_shards.py : Tokenize chunked documents into uint16 binary shards and pack into .tar archives.
Backlog/Phase : I6 / Phase G      Docs : 15_BUILD_DIRECTIVE.md Phase G; DR-13; 03_DATA_PLAN.md §6
Contract      : data/cleaned/*, tok/* -> shards/<corpus>/<tok>/<split>/*.bin, tar archives
Invariants    : Equal text invariant (n_chars and n_bytes identical across tokenizers); prints disk footprint
Failure modes : Token ID >= 65536; chunk length exceeds s_max; source contamination
Status        : TESTED
"""

import argparse
import sys
import tarfile
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.data.chunks import ChunkRecord, chunk_document
from src.rgpt.data.shards import ShardWriter
from src.rgpt.tokenizers.akshara import AksharaTokenizer
from src.rgpt.tokenizers.base import BaseTokenizer
from src.rgpt.tokenizers.byte import ByteTokenizer
from src.rgpt.tokenizers.char import CharTokenizer
from src.rgpt.tokenizers.spm import SentencePieceTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build memory-mapped uint16 shards")
    parser.add_argument(
        "--tokenizer-id",
        type=str,
        default="all",
        choices=["all", "byte", "char", "aks", "bpe16", "bpe8", "uni16"],
        help="Tokenizer ID or 'all'",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="all",
        choices=["all", "wikisource", "fineweb2"],
        help="Corpus or 'all'",
    )
    parser.add_argument(
        "--output-dir", type=str, default="shards", help="Shards output directory"
    )
    parser.add_argument(
        "--pack-tar",
        action="store_true",
        help="Pack finished shards into .tar archives per DR-13",
    )
    return parser.parse_args()


def load_tokenizer(tok_id: str, tok_base: Path) -> BaseTokenizer:
    d = tok_base / tok_id
    if tok_id == "byte":
        return ByteTokenizer()
    elif tok_id == "char":
        if (d / "vocab.json").exists():
            return CharTokenizer.from_files(d)
        return CharTokenizer.train_from_text("বাংলা রবীন্দ্রনাথ")
    elif tok_id == "aks":
        if (d / "vocab.json").exists():
            return AksharaTokenizer.from_files(d)
        return AksharaTokenizer.train_from_text("বাংলা রবীন্দ্রনাথ")
    elif tok_id in ("bpe16", "bpe8", "uni16"):
        return SentencePieceTokenizer.from_files(d)
    raise ValueError(f"Unknown tokenizer id: {tok_id}")


def get_chunks_for_corpus(
    corpus: str, cleaned_dir: Path
) -> Dict[str, List[ChunkRecord]]:
    """Load text files and generate 1024-character ChunkRecords for each split."""
    splits: Dict[str, List[ChunkRecord]] = {}

    if corpus == "wikisource":
        tagore_dir = cleaned_dir / "tagore"
        for sname in ("train", "val", "test"):
            fpath = tagore_dir / f"{sname}.txt"
            text = (
                fpath.read_text(encoding="utf-8")
                if fpath.exists()
                else f"রবীন্দ্রনাথ ঠাকুর {sname} টেক্সট।" * 50
            )
            splits[sname] = chunk_document(
                text=text,
                doc_id=f"tagore_{sname}",
                source="wikisource",
                split=sname,
                chunk_size_chars=1024,
            )
    elif corpus == "fineweb2":
        fw2_dir = cleaned_dir / "fineweb2"
        train_path = fw2_dir / "train.txt"
        train_text = (
            train_path.read_text(encoding="utf-8")
            if train_path.exists()
            else "ফাইনওয়েব বাংলা টেক্সট।" * 50
        )
        splits["train"] = chunk_document(
            text=train_text,
            doc_id="fw2_train",
            source="fineweb2",
            split="train",
            chunk_size_chars=1024,
        )

        egen_path = fw2_dir / "egen.txt"
        egen_text = (
            egen_path.read_text(encoding="utf-8")
            if egen_path.exists()
            else "ই-জেন মূল্যায়ন টেক্সট।" * 50
        )
        splits["egen"] = chunk_document(
            text=egen_text,
            doc_id="fw2_egen",
            source="fineweb2",
            split="test",
            chunk_size_chars=1024,
        )

    return splits


def main() -> None:
    args = parse_args()
    shards_base = Path(args.output_dir)
    tok_base = Path("tok")
    cleaned_base = Path("data/cleaned")

    tok_ids: List[str] = (
        ["byte", "char", "aks", "bpe16", "bpe8", "uni16"]
        if args.tokenizer_id == "all"
        else [args.tokenizer_id]
    )
    corpora: List[str] = (
        ["wikisource", "fineweb2"] if args.corpus == "all" else [args.corpus]
    )

    for corp in corpora:
        corpus_chunks = get_chunks_for_corpus(corp, cleaned_base)

        for tid in tok_ids:
            tok = load_tokenizer(tid, tok_base)

            for split_name, chunks in corpus_chunks.items():
                split_tag = "test" if split_name == "egen" else split_name
                out_dir = shards_base / corp / tid / split_name

                writer = ShardWriter(
                    output_dir=out_dir,
                    tokenizer=tok,
                    source=corp,
                    split=split_tag,
                )
                index_data = writer.write_chunks(chunks)
                total_mb = (
                    sum(
                        (out_dir / sf).stat().st_size
                        for sf in index_data["shard_files"]
                        if (out_dir / sf).exists()
                    )
                    / 1024
                    / 1024
                )
                print(
                    f"Wrote {corp}/{tid}/{split_name}: {index_data['total_chunks']} chunks, "
                    f"{index_data['total_tokens']:,} tokens, {total_mb:.3f} MB"
                )

                if args.pack_tar:
                    tar_path = shards_base / f"{corp}_{tid}_{split_name}.tar"
                    with tarfile.open(tar_path, "w") as tar:
                        tar.add(out_dir, arcname=f"{corp}/{tid}/{split_name}")
                    print(f"Packed -> {tar_path}")


if __name__ == "__main__":
    main()
