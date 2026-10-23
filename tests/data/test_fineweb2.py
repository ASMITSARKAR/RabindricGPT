"""
tests/data/test_fineweb2.py : Tests for FineWeb-2 streaming and deterministic E-gen document sampling.
"""

from src.rgpt.data.fineweb2 import sample_egen_docs, stream_fineweb2_bengali


def test_egen_deterministic_sampling() -> None:
    """E-gen documents sampled from test split must be deterministic across runs."""
    mock_test_docs = [
        {"id": f"doc_{i}", "text": f"এটি একটি নমুনা বাংলা বাক্য নম্বর {i}। " * 10}
        for i in range(20)
    ]

    docs_1 = list(sample_egen_docs(target_chars=500, source_iterable=mock_test_docs))
    docs_2 = list(sample_egen_docs(target_chars=500, source_iterable=mock_test_docs))

    assert len(docs_1) > 0
    assert len(docs_1) == len(docs_2)
    for d1, d2 in zip(docs_1, docs_2):
        assert d1["id"] == d2["id"]
        assert d1["text"] == d2["text"]


def test_egen_disjoint_from_train() -> None:
    """E-gen evaluation documents must never overlap with training documents."""
    mock_train = [{"id": f"train_{i}", "text": f"প্রশিক্ষণ নথি {i} বাংলা।"} for i in range(10)]
    mock_test = [{"id": f"test_{i}", "text": f"মূল্যায়ন নথি {i} বাংলা।"} for i in range(10)]

    train_docs = list(stream_fineweb2_bengali(source_iterable=mock_train, max_bytes=10000))
    test_docs = list(sample_egen_docs(target_chars=1000, source_iterable=mock_test))

    train_ids = {d["id"] for d in train_docs}
    test_ids = {d["id"] for d in test_docs}

    assert train_ids.isdisjoint(test_ids)
