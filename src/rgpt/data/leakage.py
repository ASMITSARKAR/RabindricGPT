"""
src/rgpt/data/leakage.py : Cross-corpus containment detection and purge of pretraining documents.
Backlog/Phase : I5 / Phase F      Docs : 15_BUILD_DIRECTIVE.md Phase F; 03_DATA_PLAN.md §5.1
Contract      : fineweb_stream, target_index -> clean_stream
Invariants    : Strict flag rule: >=3 distinct target 5-grams within a 30-word sliding window.
Failure modes : Planted test passage fails to be detected and purged
Status        : TESTED
"""

import re
from typing import Any, Dict, Iterator, List, Set, Tuple


def tokenize_bengali_words(text: str) -> List[str]:
    """Tokenize Bengali words while properly handling ZWJ and ZWNJ.

    ZWJ (\\u200D) and ZWNJ (\\u200C) are first stripped from the text to preserve
    word-internal character clusters (e.g., ya-phala 'র্য', khanda-ta) before
    matching the Bengali unicode range [\\u0980-\\u09FF]+.
    """
    clean = text.replace("\u200c", "").replace("\u200d", "")
    return re.findall(r"[\u0980-\u09FF]+", clean)


class CrossCorpusLeakageDetector:
    """Detects and purges pretraining documents that overlap with held-out target corpora.

    Flag rule: >=3 distinct target 5-grams within any 30-word sliding window.
    """

    def __init__(
        self,
        target_texts: List[str],
        ngram_size: int = 5,
        window_size: int = 30,
        threshold_distinct_ngrams: int = 3,
    ) -> None:
        """Initialize leakage detector with held-out target texts (val and test sets).

        Args:
            target_texts: List of held-out text strings.
            ngram_size: Word n-gram length (default: 5 words).
            window_size: Sliding word span size (default: 30 words).
            threshold_distinct_ngrams: Minimum distinct target n-grams in span to flag (default: 3).
        """
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.threshold = threshold_distinct_ngrams

        self.target_5grams: Set[Tuple[str, ...]] = set()

        for text in target_texts:
            words = tokenize_bengali_words(text)
            if len(words) >= self.ngram_size:
                for i in range(len(words) - self.ngram_size + 1):
                    self.target_5grams.add(tuple(words[i : i + self.ngram_size]))

    def contains_leakage(self, candidate_text: str) -> bool:
        """Return True if candidate document contains >= 3 distinct target 5-grams within a 30-word span."""
        if not candidate_text:
            return False

        words = tokenize_bengali_words(candidate_text)
        if len(words) < self.ngram_size:
            return False

        # If total words < window_size, check all 5-grams directly
        if len(words) <= self.window_size:
            matched = set()
            for i in range(len(words) - self.ngram_size + 1):
                gram = tuple(words[i : i + self.ngram_size])
                if gram in self.target_5grams:
                    matched.add(gram)
                    if len(matched) >= self.threshold:
                        return True
            return False

        # Sliding window of window_size words
        for w_start in range(len(words) - self.window_size + 1):
            span_words = words[w_start : w_start + self.window_size]
            matched = set()
            for i in range(len(span_words) - self.ngram_size + 1):
                gram = tuple(span_words[i : i + self.ngram_size])
                if gram in self.target_5grams:
                    matched.add(gram)
                    if len(matched) >= self.threshold:
                        return True

        return False

    def purge_leakage(self, docs: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Filter out leaking documents from pretraining stream."""
        for doc in docs:
            text = doc.get("text", "")
            if not self.contains_leakage(text):
                yield doc
