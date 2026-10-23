"""
src/rgpt/eval/generation.py : Text generation sampling and output quality checks (ill-formed Unicode, in-vocab rate).
Backlog/Phase : I12 / Phase J      Docs : 15_BUILD_DIRECTIVE.md Phase J; DR-7; 07_EVALUATION_PLAN.md
Contract      : sample_generation(...) -> list[str]; check_generation_quality(...) -> dict
Invariants    : Uses fixed sampling seeds [42, 137, 2026], temp 0.8, top-p 0.95; identical prompts across models
Failure modes : Generated sequence exceeds max_new_chars; infinite generation loop
Status        : TESTED
"""

import re
import unicodedata
from typing import List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rgpt.tokenizers.base import BaseTokenizer


def is_ill_formed_unicode(text: str) -> bool:
    """Check if a string contains replacement characters, lonely surrogates, or unassigned code points."""
    if "\ufffd" in text:
        return True
    for ch in text:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            return True
        if unicodedata.category(ch) == "Cn":
            return True
    return False


def check_ill_formed_unicode_rate(texts: List[str]) -> float:
    """Compute fraction of generated samples containing ill-formed Unicode or invalid byte sequences."""
    if not texts:
        return 0.0
    bad_count = sum(1 for t in texts if is_ill_formed_unicode(t))
    return bad_count / len(texts)


def check_in_vocab_word_rate(texts: List[str], reference_words: Set[str]) -> float:
    """Compute fraction of generated words that appear in the training reference vocabulary."""
    if not texts or not reference_words:
        return 0.0

    total_words = 0
    in_vocab_words = 0
    # Match Bengali or alphanumeric word tokens
    word_pattern = re.compile(r"[\u0980-\u09FF\w]+")

    for text in texts:
        words = word_pattern.findall(text)
        total_words += len(words)
        in_vocab_words += sum(1 for w in words if w in reference_words)

    if total_words == 0:
        return 0.0
    return in_vocab_words / total_words


def generate_samples(
    model: nn.Module,
    tokenizer: BaseTokenizer,
    prompts: List[str],
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_chars: int = 512,
    seed: int = 42,
    device: str = "cpu",
    eos_token_id: Optional[int] = 2,
) -> List[str]:
    """Generate completions under DR-7 sampling parameters (temp 0.8, top-p 0.95)."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    torch.manual_seed(seed)
    results: List[str] = []

    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            prompt_tokens = tokenizer.encode(prompt)
            if not prompt_tokens:
                prompt_tokens = [getattr(tokenizer, "BOS_ID", 1)]

            generated_tokens = list(prompt_tokens)
            chars_generated = 0

            # Autoregressive sampling
            while chars_generated < max_new_chars:
                x = torch.tensor([generated_tokens], dtype=torch.long, device=dev)
                # Cap sequence to model's context if needed
                if hasattr(model, "cfg") and x.shape[1] > model.cfg.s_max:
                    x = x[:, -model.cfg.s_max:]

                logits = model(x)[:, -1, :]  # [1, vocab_size]

                if temperature > 0.0:
                    scaled_logits = logits / temperature
                else:
                    scaled_logits = logits

                # Top-p (nucleus) filtering
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above top_p (shift right by 1 to keep at least 1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    scaled_logits = scaled_logits.masked_fill(indices_to_remove, float("-inf"))

                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                if eos_token_id is not None and next_token == eos_token_id:
                    break

                generated_tokens.append(next_token)

                # Track generated characters
                decoded_partial = tokenizer.decode(generated_tokens[len(prompt_tokens):])
                chars_generated = len(decoded_partial)

            full_text = tokenizer.decode(generated_tokens)
            results.append(full_text)

    return results
