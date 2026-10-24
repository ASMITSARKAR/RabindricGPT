"""
scripts/audits/inspect_shard_table.py : Aggregate and display shard metrics directly from index.json files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def collect_shard_metrics(shards_dir: Path = Path("shards")) -> List[Dict[str, Any]]:
    """Scan all index.json files under shards_dir and extract summary metrics."""
    rows: List[Dict[str, Any]] = []
    for index_path in sorted(shards_dir.glob("*/*/*/index.json")):
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
            parts = index_path.relative_to(shards_dir).parts
            corpus, tokenizer, split = parts[0], parts[1], parts[2]
            total_chunks = data.get("total_chunks", 0)
            total_tokens = data.get("total_tokens", 0)
            total_chars = data.get("total_chars", 0)
            total_bytes = data.get("total_bytes", 0)
            tok_per_chunk = total_tokens / total_chunks if total_chunks > 0 else 0.0
            chars_per_tok = total_chars / total_tokens if total_tokens > 0 else 0.0
            bytes_per_tok = total_bytes / total_tokens if total_tokens > 0 else 0.0
            rows.append({
                "corpus": corpus,
                "tokenizer": tokenizer,
                "split": split,
                "vocab_size": data.get("vocab_size", 0),
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "total_chars": total_chars,
                "total_bytes": total_bytes,
                "tokens_per_chunk": round(tok_per_chunk, 2),
                "chars_per_token": round(chars_per_tok, 3),
                "bytes_per_token": round(bytes_per_tok, 3),
                "num_shard_files": len(data.get("shard_files", [])),
                "index_path": str(index_path).replace("\\", "/"),
            })
        except Exception as e:
            print(f"Error reading {index_path}: {e}")
    return rows


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Format shard metrics into a Markdown table."""
    headers = [
        "Corpus", "Tokenizer", "Split", "Vocab", "Chunks",
        "Tokens", "Characters", "Bytes", "Tok/Chunk", "Chars/Tok"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        row_str = (
            f"| {r['corpus']} | {r['tokenizer']} | {r['split']} | {r['vocab_size']:,} | "
            f"{r['total_chunks']:,} | {r['total_tokens']:,} | {r['total_chars']:,} | "
            f"{r['total_bytes']:,} | {r['tokens_per_chunk']:.1f} | {r['chars_per_token']:.2f} |"
        )
        lines.append(row_str)
    return "\n".join(lines)


def main() -> None:
    rows = collect_shard_metrics()
    table = format_markdown_table(rows)
    print(table)


if __name__ == "__main__":
    main()
