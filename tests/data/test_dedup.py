"""
tests/data/test_dedup.py : Tests for exact SHA256 deduplication and disk-backed MinHash.
"""

from src.rgpt.data.dedup import ExactDeduplicator, MinHashDeduplicator


def test_exact_dedup_eliminates_duplicates() -> None:
    """Exact duplicate text strings must be filtered out."""
    dedup = ExactDeduplicator()
    doc1 = "রবীন্দ্রনাথ ঠাকুর ১৮৬১ সালে কলকাতায় জন্মগ্রহণ করেন।"
    doc2 = "রবীন্দ্রনাথ ঠাকুর ১৮৬১ সালে কলকাতায় জন্মগ্রহণ করেন।"
    doc3 = "কাজী নজরুল ইসলাম বাংলাদেশের জাতীয় কবি।"

    assert not dedup.is_duplicate(doc1)
    assert dedup.is_duplicate(doc2)
    assert not dedup.is_duplicate(doc3)


def test_minhash_near_dup_detection() -> None:
    """Documents sharing >=80% 5-gram char shingles must be flagged as near-duplicates."""
    dedup = MinHashDeduplicator(num_perm=64, ngram_size=5, jaccard_threshold=0.80)

    base = "আমাদের ছোট নদী চলে বাঁকে বাঁকে, বৈশাখ মাসে তার হাঁটুজল থাকে।"
    near_dup = "আমাদের ছোট নদী চলে বাঁকে বাঁকে, বৈশাখ মাসে তার হাঁটুজল থাকে। "  # Minor spacing difference
    distinct = "মেঘের কোলে রোদ হেসেছে, বাদল গেছে টুটি।"

    assert not dedup.is_near_duplicate(base)
    assert dedup.is_near_duplicate(near_dup)
    assert not dedup.is_near_duplicate(distinct)
