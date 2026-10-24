"""Compare SHA-256 of data/cleaned/** before and after the current normalizer.
Checks:
1. Current SHA-256 on disk.
2. SHA-256 after applying current normalize_text().
3. Difference flag (True if hashes differ, False if identical).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rgpt.text.normalize import normalize_text


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def process_file_whole(p: Path) -> tuple[str, str, int, int, bool, dict]:
    raw_bytes = p.read_bytes()
    sha_before = compute_sha256(raw_bytes)
    text = raw_bytes.decode("utf-8")
    norm_text, stats = normalize_text(text)
    norm_bytes = norm_text.encode("utf-8")
    sha_after = compute_sha256(norm_bytes)
    differs = (sha_before != sha_after)
    return sha_before, sha_after, len(raw_bytes), len(norm_bytes), differs, stats


def process_large_file_chunked(p: Path, chunk_chars: int = 500_000) -> tuple[str, str, int, int, bool, dict]:
    h_before = hashlib.sha256()
    h_after = hashlib.sha256()
    bytes_before = 0
    bytes_after = 0
    total_stats: dict[str, int] = {
        "zwj_kept": 0,
        "zwj_stripped": 0,
        "zwnj_kept": 0,
        "zwnj_stripped": 0,
        "u09f0_mapped": 0,
        "khanda_ta_canonicalized": 0,
        "control_chars_removed": 0,
    }

    with open(p, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            raw_b = chunk.encode("utf-8")
            h_before.update(raw_b)
            bytes_before += len(raw_b)

            norm_c, stats = normalize_text(chunk)
            norm_b = norm_c.encode("utf-8")
            h_after.update(norm_b)
            bytes_after += len(norm_b)

            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

    sha_before = h_before.hexdigest()
    sha_after = h_after.hexdigest()
    differs = (sha_before != sha_after)
    return sha_before, sha_after, bytes_before, bytes_after, differs, total_stats


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    files = [
        Path("data/cleaned/tagore/train.txt"),
        Path("data/cleaned/tagore/val.txt"),
        Path("data/cleaned/tagore/test.txt"),
        Path("data/cleaned/fineweb2/egen.txt"),
        Path("data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt"),
    ]

    print("=" * 80, flush=True)
    print("SHA-256 COMPARISON OF DATA/CLEANED/** BEFORE AND AFTER CURRENT NORMALIZER", flush=True)
    print("=" * 80, flush=True)

    any_differs = False

    for p in files:
        if not p.exists():
            print(f"\nFile: {p.as_posix()} [NOT FOUND]", flush=True)
            continue

        size = p.stat().st_size
        print(f"\nProcessing {p.as_posix()} ({size:,} bytes)...", flush=True)
        if size > 10_000_000:
            sha_before, sha_after, b_before, b_after, differs, stats = process_large_file_chunked(p)
        else:
            sha_before, sha_after, b_before, b_after, differs, stats = process_file_whole(p)

        if differs:
            any_differs = True

        print(f"  File           : {p.as_posix()}", flush=True)
        print(f"  SHA-256 before : {sha_before}", flush=True)
        print(f"  SHA-256 after  : {sha_after}", flush=True)
        print(f"  Bytes before   : {b_before:,}", flush=True)
        print(f"  Bytes after    : {b_after:,}", flush=True)
        print(f"  Differs?       : {differs}", flush=True)
        print(f"  Stats          : {stats}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"TOTAL CLEANED FILES DIFFER: {any_differs}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
