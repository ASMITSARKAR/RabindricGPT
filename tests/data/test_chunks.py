"""
tests/data/test_chunks.py : Tests for fixed-character document chunking and metadata attribution.
"""

from src.rgpt.data.chunks import chunk_document


def test_chunking_fixed_character_window() -> None:
    """Document must be sliced into chunks of at most 1024 characters per DR-1."""
    text = "বাংলা সাহিত্য " * 300  # >3000 chars
    chunks = chunk_document(
        text=text,
        doc_id="doc_1",
        source="wikisource",
        split="train",
        chunk_size_chars=1024,
    )

    assert len(chunks) > 1
    for c in chunks:
        assert c.n_chars <= 1024
        assert c.source == "wikisource"
        assert c.split == "train"

    # Concatenation of all chunk texts must exactly match original text
    reconstructed = "".join(c.text for c in chunks)
    assert reconstructed == text


def test_chunk_char_and_byte_counts() -> None:
    """Every chunk must accurately record character and exact UTF-8 byte counts."""
    text = "রবীন্দ্রনাথ ঠাকুর ছিলেন একজন নোবেলজয়ী কবি।"
    chunks = chunk_document(
        text=text,
        doc_id="doc_2",
        source="wikisource",
        split="val",
        chunk_size_chars=1024,
    )

    assert len(chunks) == 1
    c = chunks[0]
    assert c.n_chars == len(text)
    assert c.n_bytes == len(text.encode("utf-8"))
