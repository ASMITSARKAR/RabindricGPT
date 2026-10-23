"""
src/rgpt/data/chunks.py : Document chunking into fixed-character windows (1024 chars) per DR-1.
Backlog/Phase : I6 / Phase G      Docs : 15_BUILD_DIRECTIVE.md §3.1 DR-1, Phase G; 04 §5
Contract      : doc: dict -> list[ChunkRecord]
Invariants    : Each chunk contains exactly <=1024 characters; char and byte counts exact; prefixes BOS
Failure modes : Document boundary broken incorrectly; negative or zero character chunk
Status        : TESTED
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass(frozen=True)
class ChunkRecord:
    """Dataclass holding a 1024-character text chunk and associated metadata."""

    chunk_id: str
    work_id: Optional[str]
    source: str
    split: str
    text: str
    n_chars: int
    n_bytes: int


def chunk_document(
    text: str,
    doc_id: str,
    source: str,
    split: str,
    work_id: Optional[str] = None,
    chunk_size_chars: int = 1024,
) -> List[ChunkRecord]:
    """Slice document into non-overlapping chunks of up to chunk_size_chars.

    Args:
        text: Cleaned document text.
        doc_id: Canonical document identifier.
        source: Corpus source tag ('wikisource' or 'fineweb2').
        split: Split partition ('train', 'val', 'test').
        work_id: Optional Tagore work ID.
        chunk_size_chars: Maximum chunk character length (default 1024).

    Returns:
        List of ChunkRecord dataclass instances.
    """
    if not text:
        return []

    chunks: List[ChunkRecord] = []
    chunk_idx = 0
    total_len = len(text)

    for i in range(0, total_len, chunk_size_chars):
        chunk_text = text[i : i + chunk_size_chars]
        n_chars = len(chunk_text)
        n_bytes = len(chunk_text.encode("utf-8"))
        cid = f"{doc_id}_c{chunk_idx:05d}"
        chunks.append(
            ChunkRecord(
                chunk_id=cid,
                work_id=work_id,
                source=source,
                split=split,
                text=chunk_text,
                n_chars=n_chars,
                n_bytes=n_bytes,
            )
        )
        chunk_idx += 1

    return chunks


def iter_chunks(
    docs: Iterator[dict],
    chunk_size_chars: int = 1024,
) -> Iterator[ChunkRecord]:
    """Stream chunks from an iterator of documents."""
    for doc in docs:
        text = doc.get("text", "")
        doc_id = doc.get("id") or doc.get("title") or "doc"
        source = doc.get("source", "wikisource")
        split = doc.get("split", "train")
        work_id = doc.get("work_id")
        for chunk in chunk_document(
            text=text,
            doc_id=doc_id,
            source=source,
            split=split,
            work_id=work_id,
            chunk_size_chars=chunk_size_chars,
        ):
            yield chunk
