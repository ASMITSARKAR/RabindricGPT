"""
scripts/export_akshara_gold_tsv.py : Export an unlabelled stratified 500-word Akshara gold TSV for owner hand-labeling.
Ref: Owner Action T1; 04_TOKENIZER_PLAN.md §4.
"""

import re
from collections import defaultdict
from pathlib import Path
import random

def is_pure_bengali(word: str) -> bool:
    """Check if word consists strictly of Bengali script characters and optional joiners."""
    return bool(re.fullmatch(r"[\u0980-\u09FF\u200C\u200D]+", word))

def main() -> None:
    random.seed(42)
    corpus_file = Path("data/train.txt")
    if not corpus_file.exists():
        corpus_file = Path("data/raw/wikisource/pages.jsonl")

    text = corpus_file.read_text(encoding="utf-8")
    words = set(re.findall(r"[\u0980-\u09FF\u200C\u200D]+", text))
    
    # Filter out single-letter matras, invalid forms, and numerals
    valid_words = [
        w for w in words
        if len(w) >= 1
        and not re.match(r"^[\u09BE-\u09CD\u09D7]", w)
        and not re.search(r"[\u09E6-\u09EF0-9]", w)
    ]

    # Stratify into 6 distinct linguistic categories
    bins = {
        "short_particles": [],       # len 1-2
        "simple_cv_cvc": [],         # len 3-4, no conjuncts
        "single_conjunct": [],       # 1 virama
        "complex_conjunct": [],      # >=2 viramas
        "special_diacritics": [],    # candrabindu, anusvara, visarga, khanda-ta
        "vocalic_r": [],             # rri or rri-kar
    }

    for w in sorted(valid_words):
        has_virama_count = w.count('\u09CD')
        has_special = any(c in w for c in ('\u0981', '\u0982', '\u0983', '\u09CE'))
        has_rri = ('\u098B' in w or '\u09C3' in w)

        if has_rri:
            bins["vocalic_r"].append(w)
        elif has_special:
            bins["special_diacritics"].append(w)
        elif has_virama_count >= 2:
            bins["complex_conjunct"].append(w)
        elif has_virama_count == 1:
            bins["single_conjunct"].append(w)
        elif len(w) <= 2:
            bins["short_particles"].append(w)
        elif 3 <= len(w) <= 5:
            bins["simple_cv_cvc"].append(w)

    targets = {
        "short_particles": 50,
        "simple_cv_cvc": 100,
        "single_conjunct": 125,
        "complex_conjunct": 100,
        "special_diacritics": 75,
        "vocalic_r": 50,
    }

    selected = []
    for cat, target_k in targets.items():
        pool = bins[cat]
        if len(pool) < target_k:
            chosen = pool
        else:
            chosen = random.sample(pool, target_k)
        selected.extend(chosen)
        print(f"Stratum '{cat}': pool={len(pool)}, selected={len(chosen)}")

    # If any shortfall, top up from remaining valid words
    if len(selected) < 500:
        remaining = list(set(valid_words) - set(selected))
        needed = 500 - len(selected)
        selected.extend(random.sample(remaining, needed))

    selected = selected[:500]
    # Shuffle so strata are interleaved
    random.shuffle(selected)

    out_file = Path("tests/data/akshara_gold.tsv")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("word\taksharas\n")
        for w in selected:
            f.write(f"{w}\t\n")

    print(f"Wrote {len(selected)} unlabelled stratified words to {out_file}")

if __name__ == "__main__":
    main()
