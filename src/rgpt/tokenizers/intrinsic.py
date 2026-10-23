"""
src/rgpt/tokenizers/intrinsic.py : Intrinsic tokenizer evaluation metrics (T bytes/token, fertility, fallback rate, utilization).
Backlog/Phase : I3 / Phase E      Docs : 15_BUILD_DIRECTIVE.md Phase E; 04_TOKENIZER_PLAN.md §7
Contract      : tokenizer, eval_text: str -> metrics: dict[str, float]
Invariants    : T = utf8_bytes / tokens; chars_per_token = len(text) / tokens; identical text across tokenizers
Failure modes : Zero tokens produced; empty input text
Status        : TESTED
"""

from typing import Dict

from src.rgpt.tokenizers.base import BaseTokenizer


def compute_intrinsic_metrics(
    tokenizer: BaseTokenizer,
    text: str,
    top_k_vocab: int = 50000,
) -> Dict[str, float]:
    """Compute standard intrinsic metrics on eval text per 04_TOKENIZER_PLAN.md §7.

    Metrics:
        - bytes_per_token (T): UTF-8 bytes / tokens
        - chars_per_token: character count / tokens
        - fertility: tokens per whitespace word
        - fallback_rate: fraction of tokens that are byte fallbacks (IDs 3..258 or 4..259)
        - vocab_utilization: fraction of total vocab seen >= 1x
        - unk_rate: fraction of tokens that are UNK

    Args:
        tokenizer: Evaluated tokenizer instance.
        text: Evaluation text corpus.
        top_k_vocab: Unused parameter kept for API signature compatibility.

    Returns:
        Dictionary of computed metric values.
    """
    if not text:
        return {
            "bytes_per_token": 0.0,
            "chars_per_token": 0.0,
            "fertility": 0.0,
            "fallback_rate": 0.0,
            "vocab_utilization": 0.0,
            "unk_rate": 0.0,
        }

    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    if num_tokens == 0:
        return {
            "bytes_per_token": 0.0,
            "chars_per_token": 0.0,
            "fertility": 0.0,
            "fallback_rate": 0.0,
            "vocab_utilization": 0.0,
            "unk_rate": 0.0,
        }

    utf8_bytes = len(text.encode("utf-8"))
    num_chars = len(text)
    words = text.split()
    num_words = len(words) if words else 1

    # Unique tokens seen
    unique_tokens = set(tokens)
    vocab_size = tokenizer.vocab_size

    # Check fallbacks (either byte tokenizer or byte fallback range 4..259)
    fallback_count = sum(1 for tid in tokens if (4 <= tid < 260 and hasattr(tokenizer, "BYTE_FALLBACK_OFFSET")))
    unk_count = sum(1 for tid in tokens if tid == tokenizer.UNK_ID)

    return {
        "bytes_per_token": utf8_bytes / num_tokens,
        "chars_per_token": num_chars / num_tokens,
        "fertility": num_tokens / num_words,
        "fallback_rate": fallback_count / num_tokens,
        "vocab_utilization": len(unique_tokens) / vocab_size if vocab_size > 0 else 0.0,
        "unk_rate": unk_count / num_tokens,
    }
