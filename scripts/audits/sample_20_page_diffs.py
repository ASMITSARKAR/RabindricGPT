"""
scripts/audits/sample_20_page_diffs.py : Sample 20 random pages per work for before/after cleaning diffs.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.rgpt.data.wikitext_clean import clean_wikitext


def main() -> None:
    pages_by_work = defaultdict(list)
    with open("data/raw/wikisource/pages.jsonl", "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            pfx = row.get("prefix", "unknown")
            pages_by_work[pfx].append((idx, row))

    random.seed(42)
    work_summaries = []

    for work_name, pages in sorted(pages_by_work.items()):
        sample_size = min(20, len(pages))
        sampled = random.sample(pages, sample_size)
        total_raw = 0
        total_clean = 0
        removed_templates = 0
        removed_colons = 0

        for p_idx, page in sampled:
            raw = page.get("raw_text", "")
            clean, stats = clean_wikitext(raw)
            total_raw += len(raw)
            total_clean += len(clean)
            if "{{" in raw:
                removed_templates += raw.count("{{")
            if "::" in raw:
                removed_colons += raw.count("::")

        pct_removed = (1.0 - (total_clean / total_raw)) * 100 if total_raw > 0 else 0.0
        work_summaries.append({
            "work": work_name,
            "sampled_pages": sample_size,
            "total_raw_chars": total_raw,
            "total_clean_chars": total_clean,
            "pct_removed": round(pct_removed, 2),
            "templates_stripped": removed_templates,
            "colons_stripped": removed_colons,
        })

    print(json.dumps(work_summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
