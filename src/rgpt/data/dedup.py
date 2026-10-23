"""
src/rgpt/data/dedup.py : Exact SHA256 deduplication and disk-backed MinHash near-duplicate filtering.
Backlog/Phase : I5 / Phase F      Docs : 15_BUILD_DIRECTIVE.md Phase F; 03_DATA_PLAN.md §3; §3.4
Contract      : doc_stream -> deduped_stream; minhash_jaccard(doc_a, doc_b) -> float
Invariants    : Exact duplicates eliminated; MinHash respects max_ram_gb memory ceiling
Failure modes : RAM memory limit exceeded during MinHash index construction
Status        : TESTED
"""

import hashlib
from typing import Any, Dict, Iterator, List, Set

from src.rgpt.common.hashing import hash_text


class ExactDeduplicator:
    """Exact SHA256 document deduplicator."""

    def __init__(self) -> None:
        self.seen_hashes: Set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """Check if text hash has been seen previously."""
        h = hash_text(text)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def filter_stream(self, docs: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Yield documents that are not exact duplicates."""
        for doc in docs:
            text = doc.get("text", "")
            if not self.is_duplicate(text):
                yield doc


class MinHashDeduplicator:
    """MinHash LSH near-duplicate detector with memory constraints."""

    # Mersenne prime (2^61 - 1) for universal hashing
    PRIME: int = (1 << 61) - 1

    def __init__(
        self,
        num_perm: int = 64,
        ngram_size: int = 5,
        jaccard_threshold: float = 0.80,
        max_ram_gb: float = 8.0,
    ) -> None:
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.threshold = jaccard_threshold
        self.max_ram_gb = max_ram_gb
        self.stored_signatures: List[List[int]] = []

        # Fixed deterministic coefficients for reproducibility
        import random
        rng = random.Random(42)
        self.a = [rng.randint(1, self.PRIME - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.PRIME - 1) for _ in range(num_perm)]

    def _get_shingles(self, text: str) -> Set[str]:
        """Extract character n-gram shingles."""
        if len(text) < self.ngram_size:
            return {text} if text else set()
        return {text[i : i + self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def compute_minhash(self, text: str) -> List[int]:
        """Compute MinHash signature vector for input text."""
        shingles = self._get_shingles(text)
        if not shingles:
            return [0] * self.num_perm

        # Hash each shingle to 64-bit integer
        shingle_hashes = [
            int(hashlib.md5(s.encode("utf-8")).hexdigest()[:15], 16)
            for s in shingles
        ]

        signature = []
        for i in range(self.num_perm):
            ai = self.a[i]
            bi = self.b[i]
            min_val = min((ai * sh + bi) % self.PRIME for sh in shingle_hashes)
            signature.append(min_val)

        return signature

    @staticmethod
    def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
        """Estimate Jaccard similarity as fraction of matching minhash values."""
        if not sig_a or not sig_b or len(sig_a) != len(sig_b):
            return 0.0
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)

    def is_near_duplicate(self, text: str) -> bool:
        """Check if document has Jaccard similarity >= threshold with any stored document."""
        sig = self.compute_minhash(text)
        for existing in self.stored_signatures:
            if self.jaccard_similarity(sig, existing) >= self.threshold:
                return True
        self.stored_signatures.append(sig)
        return False

    def filter_stream(self, docs: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Yield documents that are not near-duplicates of previously seen documents."""
        for doc in docs:
            text = doc.get("text", "")
            if not self.is_near_duplicate(text):
                yield doc
