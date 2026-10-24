"""Audit raw joiner counts across raw input datasets before normalization.

Scans:
- data/raw/wikisource/pages.jsonl (using `raw_text`)
- data/raw/fineweb2/egen.jsonl (using `text`)
- data/raw/fineweb2/train.jsonl (using `text`)

Tracks standard and legacy joiner patterns, prints the full list of ZWJ words,
and a 40-word sample of ZWNJ words with raw/normalized code points and visual form change evaluation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rgpt.text.normalize import normalize_text

CONSONANTS_NO_RA = r"[\u0995-\u09af\u09b2\u09b6-\u09b9\u09dc-\u09df]"

PATTERNS = {
    "র+্+ZWJ+য": re.compile(r"\u09b0\u09cd\u200d\u09af"),
    "র+ZWJ+্+য (legacy)": re.compile(r"\u09b0\u200d\u09cd\u09af"),
    "র+্+ZWNJ+য": re.compile(r"\u09b0\u09cd\u200c\u09af"),
    "র+ZWNJ+্+য (legacy)": re.compile(r"\u09b0\u200c\u09cd\u09af"),
    "র+্+য": re.compile(r"\u09b0\u09cd(?!\u200d|\u200c)\u09af"),
    "other C+্+ZWJ+য": re.compile(rf"{CONSONANTS_NO_RA}\u09cd\u200d\u09af"),
    "other C+ZWJ+্+য (legacy)": re.compile(rf"{CONSONANTS_NO_RA}\u200d\u09cd\u09af"),
    "ত+্+ZWJ": re.compile(r"\u09a4\u09cd\u200d"),
    "ত+ZWJ+্ (legacy)": re.compile(r"\u09a4\u200d\u09cd"),
    "ত+্+ZWNJ": re.compile(r"\u09a4\u09cd\u200c"),
    "ত+ZWNJ+্ (legacy)": re.compile(r"\u09a4\u200c\u09cd"),
    "total ZWJ": re.compile(r"\u200d"),
    "total ZWNJ": re.compile(r"\u200c"),
}


def format_codepoints(s: str) -> str:
    return " ".join(f"U+{ord(c):04X}" for c in s)


def extract_word_at(text: str, idx: int) -> str:
    # Delimit word by whitespace or common Bengali punctuation
    delims = set(" \t\n\r।,;:\'\"!?()[]{}<>-/\\*")
    start = idx
    while start > 0 and text[start - 1] not in delims:
        start -= 1
    end = idx
    while end < len(text) and text[end] not in delims:
        end += 1
    return text[start:end]


def audit_file(path: Path, field: str) -> tuple[dict[str, int], dict[str, int], int, dict[str, int], int]:
    counts = {name: 0 for name in PATTERNS}
    total_docs = 0
    unique_zwj: dict[str, int] = {}
    unique_zwnj: dict[str, int] = {}
    total_zwj = 0
    total_zwnj = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_docs += 1
            data = json.loads(line)
            text = data.get(field, "")

            for name, pat in PATTERNS.items():
                counts[name] += len(pat.findall(text))

            # Stream extract ZWJ words
            for m in re.finditer(r"\u200d", text):
                w = extract_word_at(text, m.start())
                unique_zwj[w] = unique_zwj.get(w, 0) + 1
                total_zwj += 1

            # Stream extract ZWNJ words
            for m in re.finditer(r"\u200c", text):
                w = extract_word_at(text, m.start())
                unique_zwnj[w] = unique_zwnj.get(w, 0) + 1
                total_zwnj += 1

    counts["total_docs"] = total_docs
    return counts, unique_zwj, total_zwj, unique_zwnj, total_zwnj


def analyze_form_change(raw_word: str, norm_word: str) -> tuple[bool, str]:
    """Determine if stripping joiners changes visual rendering/graphemic form.
    
    In Bengali typography:
    1. C + virama + ZWNJ + C prevents ligature formation, showing explicit visible hasanta (e.g. চল্\u200cবে vs চলবে).
       When ZWNJ is stripped, if C+virama+C forms an orthographic conjunct (ligature), the visual form changes.
       If at word boundary (e.g. কোন্‌ -> কোন্), hasanta is always visible, visual form identical.
    2. র + virama + ZWJ + য or র + ZWJ + virama + য forces ya-phala (র্য).
       Without ZWJ, র + virama + য forms reph (র্য).
       This is a clear visual form change (reph vs ya-phala).
    3. C + virama + ZWJ + C was historically used by some OCR/typesetters for ligature control.
    """
    if raw_word == norm_word:
        return False, "No change"

    # Check for legacy ra+ya-phala
    if ("\u09b0\u09cd\u200d\u09af" in raw_word or "\u09b0\u200d\u09cd\u09af" in raw_word):
        if "\u09b0\u09cd\u09af" in norm_word:
            return True, "Form changed: ya-phala (র্য) -> reph (র্য)"
        elif "\u09b0\u09cd\u200d\u09af" in norm_word:
            return False, "Form preserved: kept ZWJ for ya-phala (র্য)"

    # Check for internal virama + ZWNJ followed by consonant
    # E.g. ক্\u200cত vs ক্ত
    internal_zwnj_conjunct = re.search(r"[\u0995-\u09b9\u09dc-\u09df]\u09cd\u200c[\u0995-\u09b9\u09dc-\u09df]", raw_word)
    if internal_zwnj_conjunct:
        return True, "Form changed: explicit visible hasanta dissolved into conjunct ligature"

    # Internal ZWJ between consonants (e.g. দেখ\u200d্তে)
    internal_zwj_ligature = re.search(r"[\u0995-\u09b9\u09dc-\u09df]\u200d[\u0995-\u09b9\u09dc-\u09df]", raw_word)
    if internal_zwj_ligature:
        return True, "Form changed: ZWJ ligature request stripped"

    # Word-final hasanta + ZWNJ (e.g. কোন্‌ -> কোন্)
    if re.search(r"\u09cd\u200c$", raw_word.rstrip("।,\"\':;!?")):
        return False, "Form preserved: word-final hasanta displays identically with or without ZWNJ"

    # ZWJ after vowel or between characters with no ligature effect
    return True, "Form changed: joiner stripped"


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    files = [
        (Path("data/raw/wikisource/pages.jsonl"), "raw_text"),
        (Path("data/raw/fineweb2/egen.jsonl"), "text"),
        (Path("data/raw/fineweb2/train.jsonl"), "text"),
    ]

    print("=" * 80)
    print("RAW JOINER AUDIT (INCLUDING LEGACY ORDERS)")
    print("=" * 80)

    for path, field in files:
        if not path.exists():
            print(f"\nFile not found: {path}")
            continue

        counts, unique_zwj, total_zwj, unique_zwnj, total_zwnj = audit_file(path, field)
        print(f"\nFile: {path.as_posix()} (field: '{field}', docs: {counts['total_docs']:,})")
        print("-" * 60)
        for pattern_name in PATTERNS:
            print(f"  {pattern_name:<26}: {counts[pattern_name]:>8,}")

        if "wikisource" in path.as_posix():
            print("\n" + "=" * 80)
            print(f"FULL UNIQUE ZWJ WORD LIST IN {path.name} (Total occurrences: {total_zwj})")
            print("=" * 80)

            zwj_form_changed_count = 0
            for idx, (raw_w, freq) in enumerate(sorted(unique_zwj.items()), 1):
                norm_w, _ = normalize_text(raw_w)
                changed, reason = analyze_form_change(raw_w, norm_w)
                if changed:
                    zwj_form_changed_count += freq
                status = "CHANGED" if changed else "UNCHANGED"
                print(f"[{idx:02d}] Freq: {freq:2d} | Status: {status}")
                print(f"     Raw : '{raw_w}' -> Code points: {format_codepoints(raw_w)}")
                print(f"     Norm: '{norm_w}' -> Code points: {format_codepoints(norm_w)}")
                print(f"     Note: {reason}")

            print("\n" + "-" * 80)
            print(f"ZWJ Summary: {len(unique_zwj)} unique words, {total_zwj} total occurrences.")
            print(f"Words that changed form: {zwj_form_changed_count} occurrences.")
            print("-" * 80)

            print("\n" + "=" * 80)
            print(f"SAMPLE 40 ZWNJ WORDS IN {path.name} (Total occurrences: {total_zwnj})")
            print("=" * 80)

            sample_40 = list(sorted(unique_zwnj.items(), key=lambda x: -x[1]))[:40]
            zwnj_form_changed_sample = 0
            for idx, (raw_w, freq) in enumerate(sample_40, 1):
                norm_w, _ = normalize_text(raw_w)
                changed, reason = analyze_form_change(raw_w, norm_w)
                if changed:
                    zwnj_form_changed_sample += freq
                status = "CHANGED" if changed else "UNCHANGED"
                print(f"[{idx:02d}] Freq: {freq:3d} | Status: {status}")
                print(f"     Raw : '{raw_w}' -> Code points: {format_codepoints(raw_w)}")
                print(f"     Norm: '{norm_w}' -> Code points: {format_codepoints(norm_w)}")
                print(f"     Note: {reason}")

            print("\n" + "-" * 80)
            print(f"Total unique ZWNJ words in corpus: {len(unique_zwnj)}")
            print(f"In top 40 sample ({sum(f for _, f in sample_40)} occurrences), changed form: {zwnj_form_changed_sample} occurrences.")
            print("-" * 80)


if __name__ == "__main__":
    main()
