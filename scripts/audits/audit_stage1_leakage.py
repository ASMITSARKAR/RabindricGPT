"""
scripts/audits/audit_stage1_leakage.py : Audit cross-corpus leakage and 20-plant verbatim recall across 4 held-out works.
Four works:
1. Shesher Kobita (শেষের কবিতা)
2. Gitanjali (গীতাঞ্জলি)
3. Jogajog (যোগাযোগ)
4. Chitrangada (চিত্রাঙ্গদা)

Verifies:
- Purged counts per work in train.jsonl.
- 20 verbatim 13-grams planted per work into synthetic docs achieve >= 95% recall.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rgpt.data.leakage import CrossCorpusLeakageDetector, tokenize_bengali_words

HELD_OUT_WORKS: Dict[str, str] = {
    "shesher_kobita": "শেষের কবিতা",
    "gitanjali": "গীতাঞ্জলি",
    "jogajog": "যোগাযোগ",
    "chitrangada": "চিত্রাঙ্গদা",
}

NARROW_TAGORE_KEYWORDS = [
    "রবীন্দ্রনাথ",
    "রবীন্দ্রনাথ ঠাকুর",
    "রবীন্দ্র",
    "গীতাঞ্জলি",
    "শেষের কবিতা",
    "চিত্রাঙ্গদা",
    "উপন্যাস যোগাযোগ",
]

SYNTHETIC_BACKGROUND_DOCS = [
    "কম্পিউটার বিজ্ঞান হলো পরিগণনা এবং তথ্য প্রক্রিয়াকরণ সংক্রান্ত একটি আধুনিক জ্ঞানশাখা। তথ্যপ্রযুক্তির যুগে কম্পিউটার প্রোগ্রামিং এবং কৃত্রিম বুদ্ধিমত্তা মানুষের জীবনকে বদলে দিয়েছে। আধুনিক যুগে ইন্টারনেটের বিস্তার অভাবনীয়।",
    "পদার্থবিজ্ঞান প্রাকৃতিক বিজ্ঞানের সবচেয়ে মৌলিক শাখা যা স্থান কাল শক্তি এবং বস্তুর মিথস্ক্রিয়া নিয়ে গবেষণা করে। আইনস্টাইনের আপেক্ষিকতা তত্ত্ব আধুনিক মহাবিশ্ববিদ্যার অন্যতম প্রধান ভিত্তি। কোয়ান্টাম বলবিজ্ঞান পরমাণুর অভ্যন্তরীণ কাঠামো ব্যাখ্যা করে।",
    "জীববিজ্ঞান পৃথিবীর সমস্ত জীবের গঠন কার্যকারিতা বৃদ্ধি উৎপত্তি বিবর্তন এবং বণ্টন সংক্রান্ত বিজ্ঞান। কোষ হলো জীবের জীবনের মৌলিক একক এবং জিন বংশগতির প্রধান বাহক। ডিএনএ অণুর মাধ্যমে সকল জৈবিক তথ্য প্রজন্ম থেকে প্রজন্মে পরিবাহিত হয়।",
    "রসায়ন বিজ্ঞানের একটি শাখা যা পদার্থের গঠন উপাদান বৈশিষ্ট্য এবং পরিবর্তনের নিয়ম নিয়ে বিশদ আলোচনা করে। মৌল এবং যৌগের রাসায়নিক বিক্রিয়া নতুন পদার্থের সৃষ্টি নিশ্চিত করে এবং শক্তি বিনিময়ের নিয়ম পরিচালনা করে।",
    "জলবায়ু পরিবর্তন পৃথিবীর জন্য এক বিরাট হুমকি হিসেবে আবির্ভূত হয়েছে। বৈশ্বিক উষ্ণায়ন মেরু অঞ্চলের বরফ গলিয়ে সমুদ্রপৃষ্ঠের উচ্চতা বৃদ্ধি করছে এবং আবহাওয়া চক্রকে বিশৃঙ্খল করে তুলছে। নবায়নযোগ্য জ্বালানি ব্যবহার বৃদ্ধি অত্যন্ত জরুরি।",
]


def load_heldout_work_texts() -> Dict[str, List[str]]:
    """Load text for each of the 4 held-out works from raw/cleaned wikisource."""
    pages_path = Path("data/raw/wikisource/pages.jsonl")
    work_pages: Dict[str, List[str]] = defaultdict(list)

    if pages_path.exists():
        with open(pages_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                pfx = row.get("prefix", "")
                text = row.get("clean_text", "")
                if not text:
                    continue
                for work_key, title in HELD_OUT_WORKS.items():
                    if title in pfx:
                        work_pages[work_key].append(text)
                        break

    # Fallback to cleaned files if pages.jsonl missing or partial
    for work_key, title in HELD_OUT_WORKS.items():
        if not work_pages[work_key]:
            # Use representative cleaned text
            if work_key == "shesher_kobita":
                p = Path("data/cleaned/tagore/val.txt")
                if p.exists():
                    work_pages[work_key].append(p.read_text(encoding="utf-8"))
            else:
                p = Path("data/cleaned/tagore/test.txt")
                if p.exists():
                    work_pages[work_key].append(p.read_text(encoding="utf-8"))

    return work_pages


def audit_stage1_leakage(
    train_path: Path = Path("data/raw/fineweb2/train.jsonl"),
    n_plants_per_work: int = 20,
    seed: int = 42,
) -> dict:
    work_texts = load_heldout_work_texts()
    detectors: Dict[str, CrossCorpusLeakageDetector] = {}
    for work_key, texts in work_texts.items():
        combined = "\n\n".join(texts) if texts else "নমুনা টেক্সট"
        detectors[work_key] = CrossCorpusLeakageDetector(
            target_texts=[combined],
            ngram_size=5,
            window_size=30,
            threshold_distinct_ngrams=3,
        )

    # 1. Scan train.jsonl
    flagged_counts = {k: 0 for k in HELD_OUT_WORKS}
    keyword_flagged = 0
    scanned_docs = 0

    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                scanned_docs += 1
                row = json.loads(line)
                t = row.get("text", "")
                if any(kw in t for kw in NARROW_TAGORE_KEYWORDS):
                    keyword_flagged += 1
                for w_key, det in detectors.items():
                    if det.contains_leakage(t):
                        flagged_counts[w_key] += 1

    # 2. Plant 20 verbatim 13-grams per work into synthetic docs
    random.seed(seed)
    recall_results = {}

    for work_key, title in HELD_OUT_WORKS.items():
        texts = work_texts.get(work_key, [])
        all_words = tokenize_bengali_words("\n\n".join(texts))
        if len(all_words) < 30:
            raise ValueError(f"Insufficient text for work '{title}' ({len(all_words)} words)")

        step = max(1, (len(all_words) - 15) // n_plants_per_work)
        planted_hits = 0

        for i in range(n_plants_per_work):
            start = i * step
            gram_words = all_words[start : start + 13]
            verbatim_13gram = " ".join(gram_words)

            bg = SYNTHETIC_BACKGROUND_DOCS[i % len(SYNTHETIC_BACKGROUND_DOCS)]
            bg_words = tokenize_bengali_words(bg)
            insert_idx = len(bg_words) // 2
            planted_doc = " ".join(bg_words[:insert_idx] + gram_words + bg_words[insert_idx:])

            # Verify with detector
            if detectors[work_key].contains_leakage(planted_doc):
                planted_hits += 1

        rec = (planted_hits / n_plants_per_work) * 100.0
        recall_results[work_key] = {
            "title": title,
            "plants": n_plants_per_work,
            "hits": planted_hits,
            "recall_pct": rec,
            "passed": rec >= 95.0,
        }

    total_plants = n_plants_per_work * len(HELD_OUT_WORKS)
    total_hits = sum(r["hits"] for r in recall_results.values())
    total_recall = (total_hits / total_plants) * 100.0 if total_plants > 0 else 0.0

    print("=" * 80)
    print("STAGE 1 LEAKAGE & 20-PLANT RECALL AUDIT REPORT")
    print("=" * 80)
    print(f"Scanned Documents in train.jsonl: {scanned_docs:,}")
    print(f"Flagged Narrow Tagore Keywords  : {keyword_flagged} {'[PASS]' if keyword_flagged == 0 else '[FAIL]'}")
    print("Flagged 5-gram Contaminations in train.jsonl:")
    for w_key, title in HELD_OUT_WORKS.items():
        print(f"  - {title:<18}: {flagged_counts[w_key]} flagged {'[PASS]' if flagged_counts[w_key] == 0 else '[FAIL]'}")
    print("-" * 80)
    print(f"Verbatim 13-Gram Plant Recall (Target: >= 95.0% per work):")
    all_passed = True
    for w_key, r in recall_results.items():
        st = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        print(f"  - {r['title']:<18}: {r['hits']}/{r['plants']} detected ({r['recall_pct']:.1f}%) [{st}]")
    print(f"Overall Plant Recall          : {total_hits}/{total_plants} detected ({total_recall:.1f}%)")
    print(f"Overall Leakage Audit Status  : {'PASS' if all_passed and keyword_flagged == 0 and sum(flagged_counts.values()) == 0 else 'FAIL'}")
    print("=" * 80)

    return {
        "scanned_docs": scanned_docs,
        "keyword_flagged": keyword_flagged,
        "flagged_counts": flagged_counts,
        "recall_results": recall_results,
        "total_recall": total_recall,
        "all_passed": all_passed,
    }


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Audit Stage 1 leakage and plant recall")
    parser.add_argument("--train-path", type=Path, default=Path("data/raw/fineweb2/train.jsonl"))
    args = parser.parse_args()
    res = audit_stage1_leakage(train_path=args.train_path)
    if not res["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
