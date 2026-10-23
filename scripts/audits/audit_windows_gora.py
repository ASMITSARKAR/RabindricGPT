"""
scripts/audits/audit_windows_gora.py : Trace train windows 0-99 to source works/pages,
compute compression profile across consecutive 100-window blocks of Gora,
and generate reconciled per-work compression table for all 10 works across all 6 tokenizers.
Contract: data/cleaned/tagore/train.txt, shards/wikisource/ -> results/audits/audit_windows_gora.txt
"""

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rgpt.common.jsonlog import read_jsonl
from src.rgpt.data.dedup import ExactDeduplicator
from src.rgpt.data.splits import assign_work_splits
from src.rgpt.data.wikitext_clean import clean_wikitext
from src.rgpt.tokenizers.akshara import AksharaTokenizer
from src.rgpt.tokenizers.byte import ByteTokenizer
from src.rgpt.tokenizers.char import CharTokenizer
from src.rgpt.tokenizers.spm import SentencePieceTokenizer


def get_git_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    train_txt_path = root / "data" / "cleaned" / "tagore" / "train.txt"
    pages_jsonl_path = root / "data" / "raw" / "wikisource" / "pages.jsonl"
    works_yaml_path = root / "configs" / "works_wikisource.yaml"

    works_config = yaml.safe_load(works_yaml_path.read_text(encoding="utf-8"))
    works = works_config.get("works", [])
    splits = assign_work_splits(works, split_option="Option_A")
    prefix_to_split = {
        w["prefix"]: s for s, wl in splits.items() for w in wl if "prefix" in w
    }
    id_to_split = {w["id"]: s for s, wl in splits.items() for w in wl}
    pfx_to_work = {w["prefix"]: w["id"] for w in works if "prefix" in w}

    # Load and map train pages in exact order as cleaned
    exact_dedup = ExactDeduplicator()
    page_records = []
    current_pos = 0

    for row in read_jsonl(pages_jsonl_path):
        pfx = row.get("prefix", "")
        wid = pfx_to_work.get(pfx) or row.get("work_id", "")
        target_split = prefix_to_split.get(pfx) or id_to_split.get(wid, "train")
        if target_split != "train":
            continue
        text = row.get("clean_text") or clean_wikitext(row.get("raw_text", ""))[0]
        if not exact_dedup.is_duplicate(text):
            if page_records:
                current_pos += 2
            start = current_pos
            end = start + len(text)
            current_pos = end
            page_records.append(
                {
                    "title": row.get("title", ""),
                    "work_id": wid,
                    "start": start,
                    "end": end,
                    "len": len(text),
                }
            )

    train_text = train_txt_path.read_text(encoding="utf-8")

    # Load tokenizers
    tok_bpe16 = SentencePieceTokenizer.from_files(root / "tok" / "bpe16")
    tok_bpe8 = SentencePieceTokenizer.from_files(root / "tok" / "bpe8")
    tok_uni16 = SentencePieceTokenizer.from_files(root / "tok" / "uni16")
    tok_byte = ByteTokenizer()
    tok_char = CharTokenizer.from_files(root / "tok" / "char")
    tok_aks = AksharaTokenizer.from_files(root / "tok" / "aks")

    out_lines: List[str] = []
    out_lines.append("=== AUDIT: WINDOWS 0-99 & GORA COMPRESSION PROFILE ===")
    out_lines.append(f"Git commit SHA: {get_git_sha()}")
    out_lines.append(
        f"Input train.txt SHA-256: {sha256_file(train_txt_path)} ({train_txt_path.stat().st_size} bytes)"
    )
    out_lines.append(
        f"Input pages.jsonl SHA-256: {sha256_file(pages_jsonl_path)} ({pages_jsonl_path.stat().st_size} bytes)"
    )
    out_lines.append(
        f"Input works_wikisource.yaml SHA-256: {sha256_file(works_yaml_path)}"
    )
    out_lines.append("")

    # Part 1: Windows 0-99
    out_lines.append("--- PART 1: WINDOWS 0-99 (tagore_train_c00000 - c00099) ---")
    out_lines.append(
        f"{'Window ID':<20} | {'Char Range':<15} | {'Work(s)':<25} | {'Pages':<5} | {'Chars/Tok (bpe16)':<18} | {'Sample Page':<35}"
    )
    out_lines.append("-" * 125)

    window_work_counts: Dict[str, int] = {}
    for w_idx in range(100):
        cid = f"tagore_train_c{w_idx:05d}"
        c_start = w_idx * 1024
        c_end = min(c_start + 1024, len(train_text))
        chunk_text = train_text[c_start:c_end]
        tokens = len(tok_bpe16.encode(chunk_text))
        c_per_t = len(chunk_text) / tokens if tokens else 0

        pages = [
            p
            for p in page_records
            if not (p["end"] <= c_start or p["start"] >= c_end)
        ]
        wids = sorted(list(set(p["work_id"] for p in pages)))
        for wid in wids:
            window_work_counts[wid] = window_work_counts.get(wid, 0) + 1
        sample_page = pages[0]["title"] if pages else "None"
        if len(sample_page) > 34:
            sample_page = sample_page[:31] + "..."
        out_lines.append(
            f"{cid:<20} | {f'{c_start}-{c_end}':<15} | {', '.join(wids):<25} | {len(pages):<5} | {c_per_t:<18.3f} | {sample_page:<35}"
        )

    out_lines.append("-" * 125)
    out_lines.append("Windows 0-99 Work Summary (window count):")
    for wid, count in sorted(window_work_counts.items()):
        out_lines.append(f"  - {wid}: {count} windows")
    out_lines.append("")

    # Part 2: Consecutive 100-window blocks of Gora
    gora_pages = [p for p in page_records if p["work_id"] == "gora"]
    gora_start = gora_pages[0]["start"]
    gora_end = gora_pages[-1]["end"]
    gora_text = train_text[gora_start:gora_end]
    n_gora_windows = (len(gora_text) + 1023) // 1024

    out_lines.append("--- PART 2: CONSECUTIVE 100-WINDOW BLOCKS OF GORA ---")
    out_lines.append(
        f"Gora character span in train.txt: {gora_start} to {gora_end} (length: {len(gora_text)} chars)"
    )
    out_lines.append(
        f"Gora chunk range in train shards: c{gora_start // 1024:05d} to c{gora_end // 1024:05d} (total {n_gora_windows} 1024-char windows)"
    )
    out_lines.append("")
    out_lines.append(
        f"{'Gora Window Block':<22} | {'Chars':<8} | {'bpe16 c/t':<10} | {'bpe8 c/t':<10} | {'uni16 c/t':<10} | {'char c/t':<10} | {'aks c/t':<10} | {'byte c/t':<10}"
    )
    out_lines.append("-" * 105)

    for block_idx in range((n_gora_windows + 99) // 100):
        w_start = block_idx * 100
        w_end = min((block_idx + 1) * 100, n_gora_windows)
        c_start = w_start * 1024
        c_end = min(w_end * 1024, len(gora_text))
        sub = gora_text[c_start:c_end]
        t16 = len(tok_bpe16.encode(sub))
        t8 = len(tok_bpe8.encode(sub))
        tu = len(tok_uni16.encode(sub))
        tc = len(tok_char.encode(sub))
        ta = len(tok_aks.encode(sub))
        tb = len(tok_byte.encode(sub))
        c16 = len(sub) / t16
        c8 = len(sub) / t8
        cu = len(sub) / tu
        cc = len(sub) / tc
        ca = len(sub) / ta
        cb = len(sub) / tb
        out_lines.append(
            f"Windows {w_start:03d}-{w_end - 1:03d}          | {len(sub):<8} | {c16:<10.3f} | {c8:<10.3f} | {cu:<10.3f} | {cc:<10.3f} | {ca:<10.3f} | {cb:<10.3f}"
        )

    out_lines.append("-" * 95)
    out_lines.append("")

    # Part 3: Reconciled per-work table for all 10 works
    out_lines.append("--- PART 3: RECONCILED PER-WORK TABLE (ALL 10 WORKS) ---")
    out_lines.append(
        f"{'Work ID':<16} | {'Genre':<8} | {'Split':<5} | {'Clean Chars':<11} | {'bpe16 c/t':<10} | {'bpe8 c/t':<10} | {'uni16 c/t':<10} | {'char c/t':<10} | {'aks c/t':<10} | {'byte c/t':<10}"
    )
    out_lines.append("-" * 110)

    # Group all pages across all splits by work_id
    work_texts: Dict[str, List[str]] = {w["id"]: [] for w in works}
    exact_dedup_all = ExactDeduplicator()
    for row in read_jsonl(pages_jsonl_path):
        pfx = row.get("prefix", "")
        wid = pfx_to_work.get(pfx) or row.get("work_id", "")
        if wid not in work_texts:
            continue
        text = row.get("clean_text") or clean_wikitext(row.get("raw_text", ""))[0]
        if not exact_dedup_all.is_duplicate(text):
            work_texts[wid].append(text)

    for w in works:
        wid = w["id"]
        genre = w["genre"]
        split = w["split"]
        full_work_text = "\n\n".join(work_texts[wid])
        n_chars = len(full_work_text)
        t16 = len(tok_bpe16.encode(full_work_text))
        t8 = len(tok_bpe8.encode(full_work_text))
        tu = len(tok_uni16.encode(full_work_text))
        tc = len(tok_char.encode(full_work_text))
        ta = len(tok_aks.encode(full_work_text))
        tb = len(tok_byte.encode(full_work_text))

        c16 = n_chars / t16 if t16 else 0
        c8 = n_chars / t8 if t8 else 0
        cu = n_chars / tu if tu else 0
        cc = n_chars / tc if tc else 0
        ca = n_chars / ta if ta else 0
        cb = n_chars / tb if tb else 0

        out_lines.append(
            f"{wid:<16} | {genre:<8} | {split:<5} | {n_chars:<11} | {c16:<10.3f} | {c8:<10.3f} | {cu:<10.3f} | {cc:<10.3f} | {ca:<10.3f} | {cb:<10.3f}"
        )

    out_lines.append("-" * 110)
    out_lines.append("")
    out_lines.append("Audit status: COMPLETE")

    res_txt = "\n".join(out_lines) + "\n"
    out_file = root / "results" / "audits" / "audit_windows_gora.txt"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(res_txt, encoding="utf-8")
    print(f"Wrote pinned audit output to {out_file}")


if __name__ == "__main__":
    main()
