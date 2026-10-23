"""
scripts/make_mini_corpus.py : Generate deterministic synthetic mini-corpus for offline P3 end-to-end testing.
Backlog/Phase : Phase P      Docs : 15_BUILD_DIRECTIVE.md Phase P (Level P3)
Contract      : tests/data/mini/ -> mini_wikisource.jsonl, mini_fineweb.jsonl
Invariants    : Fully deterministic via fixed random seed (42); synthetic Bengali akshara sequences
Status        : TESTED
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rgpt.common.paths import get_project_root

AKSHARAS = [
    "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ",
    "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন",
    "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ", "স", "হ",
    "কা", "কি", "কী", "কু", "কূ", "কে", "কো", "তা", "তি", "তে",
    "বা", "বি", "বে", "রা", "রি", "রে", "মা", "মি", "মে",
    "ক্ত", "ক্ষ", "ঙ্ক", "ঙ্গ", "ঞ্চ", "ঞ্জ", "ন্ট", "ন্ড", "ন্ত", "ন্দ", "ম্প", "ম্ব",
]


def generate_text(num_words: int, rng: random.Random) -> str:
    words = []
    for _ in range(num_words):
        w_len = rng.randint(2, 5)
        word = "".join(rng.choice(AKSHARAS) for _ in range(w_len))
        words.append(word)
    return " ".join(words) + "।"


def make_mini_corpus(output_dir: Path, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # 1. Wikisource works (3 works, multiple pages)
    ws_file = output_dir / "mini_wikisource.jsonl"
    with open(ws_file, "w", encoding="utf-8") as f:
        for work_idx in range(1, 4):
            work_id = f"work_{work_idx}"
            genre = "poetry" if work_idx == 1 else "prose"
            for page_idx in range(1, 4):
                text = generate_text(num_words=80, rng=rng)
                doc = {
                    "work_id": work_id,
                    "title": f"রচনাবলী {work_idx}",
                    "page_id": f"{work_id}_p{page_idx}",
                    "source": "wikisource",
                    "genre": genre,
                    "text": text,
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 2. Fineweb records (5 documents)
    fw_file = output_dir / "mini_fineweb.jsonl"
    with open(fw_file, "w", encoding="utf-8") as f:
        for doc_idx in range(1, 6):
            text = generate_text(num_words=100, rng=rng)
            doc = {
                "doc_id": f"fw_mini_{doc_idx}",
                "source": "fineweb2",
                "text": text,
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Generated synthetic mini corpus in {output_dir}")


def main() -> None:
    root = get_project_root()
    mini_dir = root / "tests" / "data" / "mini"
    make_mini_corpus(mini_dir)


if __name__ == "__main__":
    main()
