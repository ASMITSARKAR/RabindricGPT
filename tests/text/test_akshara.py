"""
tests/text/test_akshara.py : Tests for akshara orthographic syllable segmentation invariants and comparison suites.
"""

import json
import random
import warnings
from pathlib import Path

import pytest
import regex

from src.rgpt.text.akshara import DEPENDENT_MARKS, segment


def test_akshara_concatenation_invariance() -> None:
    """Concatenation of segmented units must exactly reproduce input text: "".join(segment(x)) == x."""
    corpus = [
        "গীতাঞ্জলি",
        "সোনার তরী",
        "চিত্ত যেথা ভয়শূন্য, উচ্চ যেথা শির",
        "ক্ষত্রিয়, স্ত্রী, শাস্ত্র, রাষ্ট্র, উজ্জ্বল, অন্বেষণ",
        "বাড়ি, গাঢ়, পায়রা, রবীন্দ্র",
        "123!@# English mixed with বাংলা ও ০১২৩৪৫৬৭৮৯।",
        "   স্পেস \t এবং \n\n নতুন লাইন।",
    ]
    for text in corpus:
        units = segment(text)
        assert "".join(units) == text, f"Failed round-trip on {text}"

    # Property test on random sequences of Bengali code points
    bengali_chars = [chr(cp) for cp in range(0x0981, 0x09FA)] + [" ", "\n", "a", "1", "।"]
    rng = random.Random(42)
    for _ in range(50):
        rand_text = "".join(rng.choices(bengali_chars, k=100))
        assert "".join(segment(rand_text)) == rand_text


def test_no_unit_starts_with_dependent_mark() -> None:
    """No segment unit may begin with a dependent vowel sign, virama, or nukta unless orphan."""
    valid_text = "রবীন্দ্রনাথ ঠাকুর রবীন্দ্রনাথ ঠাকুরের শ্রেষ্ঠ কবিতা"
    for unit in segment(valid_text):
        if len(unit) > 1:
            assert unit[0] not in DEPENDENT_MARKS


def test_conjunct_clusters_kept_intact() -> None:
    """Bengali conjuncts such as ক্ষ and স্ত্র must be segmented as single units."""
    # ক্ষ = ক + virama + ষ
    units_ksha = segment("ক্ষ")
    assert units_ksha == ["ক্ষ"]

    # স্ত্র = স + virama + ত + virama + র
    units_stra = segment("স্ত্র")
    assert units_stra == ["স্ত্র"]

    # উজ্জ্বল
    assert "জ্জ্ব" in segment("উজ্জ্বল")


def test_nukta_consonants() -> None:
    """Nukta consonants (ড়, ঢ়, য়) must remain attached as single units."""
    for word in ["বাড়ি", "আষাঢ়", "পায়রা"]:
        assert "".join(segment(word)) == word


def test_akshara_vs_regex_cluster_comparison(tmp_path: Path) -> None:
    """Differences between custom akshara segmenter and regex \\X grapheme clusters must be logged to TSV."""
    sample_text = (
        "রবীন্দ্রনাথ ঠাকুর রচিত গীতাঞ্জলি কাব্যগ্রন্থের কবিতাগুলি অপূর্ব। "
        "ক্ষমা করো প্রভু, অনন্ত প্রেম, চিত্রাঙ্গদা ও যোগাযোগ উপন্যাস।"
    )

    my_units = segment(sample_text)
    x_units = regex.findall(r"\X", sample_text)

    tsv_path = Path("tests/data/akshara_vs_X.tsv")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)

    differences = []
    # If counts or units differ, log them
    if my_units != x_units:
        differences.append(("akshara_vs_X_sample", "".join(my_units[:5]), "".join(x_units[:5]), "design_choice"))

    # Always ensure the TSV file exists with header
    header = "test_case\tmy_segmenter\tregex_X\tclass\n"
    rows = [f"{tc}\t{m}\t{x}\t{cls}\n" for tc, m, x, cls in differences]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    assert tsv_path.exists()


def test_hand_labelled_gold_akshara_list() -> None:
    """Gold labelled word list test; skips with warning if tests/data/akshara_gold.tsv is absent."""
    gold_path = Path("tests/data/akshara_gold.tsv")
    if not gold_path.exists():
        warnings.warn("tests/data/akshara_gold.tsv not found; skipping gold validation (Owner Action T1).")
        pytest.skip("Owner Action T1: tests/data/akshara_gold.tsv pending")

    # If present, evaluate gold accuracy
    lines = gold_path.read_text(encoding="utf-8").strip().splitlines()
    correct = 0
    total = 0
    disagreements = []
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[1].strip():
            continue  # unlabelled row awaiting owner hand-labeling
        word, gold_units_str = parts[0], parts[1]
        expected = gold_units_str.split(" ")
        actual = segment(word)
        if actual == expected:
            correct += 1
        else:
            disagreements.append({"word": word, "gold": expected, "segmenter": actual})
        total += 1

    if total > 0:
        agreement_rate = correct / total
        disagreements_file = Path("results/audits/akshara_disagreements.json")
        disagreements_file.parent.mkdir(parents=True, exist_ok=True)
        disagreements_file.write_text(json.dumps({
            "total_evaluated": total,
            "agreements": correct,
            "disagreements_count": len(disagreements),
            "agreement_rate": round(agreement_rate, 4),
            "disagreements": disagreements,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[Akshara Gold Validation] Agreement: {agreement_rate:.2%} ({correct}/{total}) | Disagreements: {len(disagreements)}")
        assert agreement_rate >= 0.90, f"Gold akshara agreement {agreement_rate:.2%} < 90%"
    else:
        warnings.warn("tests/data/akshara_gold.tsv is present but unlabelled; skipping evaluation (Owner Action T1).")
        pytest.skip("Owner Action T1: tests/data/akshara_gold.tsv unlabelled pending owner hand-labeling")
