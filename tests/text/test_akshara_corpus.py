"""
tests/text/test_akshara_corpus.py : Tests for corpus-wide akshara validation routines.
Validates:
1. Losslessness on 20-word synthetic set covering all 6 strata.
2. Each invariant rule fires on hand-built bad units and passes on valid units.
3. Wilson interval matches analytical known values.
4. Review queue writer leaves owner_verdict empty.
5. Gold set evaluation cleanly handles unlabelled files and rejects malformed rows.
"""

from collections import Counter
from pathlib import Path
import pytest

from scripts.audits.audit_akshara_corpus import (
    STRATA_NAMES,
    check_virama_last_char,
    classify_word_stratum,
    contains_consonant_after_vowel_sign,
    contains_multiple_independent_vowels,
    contains_space_or_newline,
    evaluate_gold_set,
    starts_with_dependent_mark,
    wilson_score_interval,
    write_review_queue,
)
from src.rgpt.text.akshara import segment


def test_losslessness_synthetic_six_strata() -> None:
    """Losslessness must hold across a 24-word synthetic set covering all 6 linguistic strata."""
    synthetic_corpus = {
        "short_particles": ["ও", "না", "সে", "কী"],
        "simple_cv_cvc": ["বাগান", "বাতাস", "কলম", "নদী"],
        "single_conjunct": ["রক্ত", "শব্দ", "কষ্ট", "শান্তি"],
        "complex_conjunct": ["উজ্জ্বল", "রাষ্ট্র", "শাস্ত্র", "লক্ষ্মী"],
        "special_diacritics": ["হঠাৎ", "চাঁদ", "দুঃসময়", "রং"],
        "vocalic_r": ["ঋতু", "কৃষি", "হৃদয়", "বৃষ্টি"],
    }

    total_words = 0
    for stratum_name, words in synthetic_corpus.items():
        assert stratum_name in STRATA_NAMES, f"Stratum {stratum_name} must be in STRATA_NAMES"
        for word in words:
            total_words += 1
            # Stratum classification check
            classified = classify_word_stratum(word)
            assert classified == stratum_name, f"Expected {stratum_name} for '{word}', got {classified}"
            # Losslessness invariant check: "".join(segment(w)) == w
            segmented = segment(word)
            assert "".join(segmented) == word, f"Losslessness violation on synthetic word '{word}'"

    assert total_words == 24, "Must cover at least 20 synthetic words across all six strata"


def test_invariant_rules_fire_on_hand_built_bad_units() -> None:
    """Each structural invariant rule must fire on a hand-built bad unit and pass on valid units."""
    # Rule 1: Starts with virama, dependent vowel sign, nukta, or modifier
    bad_rule1 = ["\u09cdক", "\u09be", "\u09bc", "\u0981", "\u0982", "\u0983"]
    for bad in bad_rule1:
        assert starts_with_dependent_mark(bad), f"Rule 1 failed to fire on bad unit {repr(bad)}"
    assert not starts_with_dependent_mark("ক")
    assert not starts_with_dependent_mark("কা")
    assert not starts_with_dependent_mark("আম")

    # Rule 2: Contains virama as last character (except word-final hasanta)
    # When is_word_final=False, trailing virama unit is flagged as violation
    viol, is_wf = check_virama_last_char("ক্", is_word_final=False)
    assert viol and not is_wf, "Rule 2 failed to fire on mid-word virama unit 'ক্'"

    viol, is_wf = check_virama_last_char("ন্", is_word_final=False)
    assert viol and not is_wf, "Rule 2 failed to fire on mid-word virama unit 'ন্'"

    # When is_word_final=True and has base letter, counted as word-final hasanta, not violation
    viol, is_wf = check_virama_last_char("ক্", is_word_final=True)
    assert not viol and is_wf, "Word-final hasanta 'ক্' must be classified as word-final hasanta"

    # Orphan virama is always a violation even if at word boundary
    viol, is_wf = check_virama_last_char("\u09cd", is_word_final=True)
    assert viol and not is_wf, "Orphan virama must be flagged as violation even if at boundary"

    # Clean units do not fire Rule 2
    viol, is_wf = check_virama_last_char("ক", is_word_final=False)
    assert not viol and not is_wf
    viol, is_wf = check_virama_last_char("কা", is_word_final=False)
    assert not viol and not is_wf

    # Rule 3: Contains two or more independent vowels
    bad_rule3 = ["অআ", "ইঈ", "উঊএ", "ঐঔ"]
    for bad in bad_rule3:
        assert contains_multiple_independent_vowels(bad), f"Rule 3 failed to fire on {repr(bad)}"
    assert not contains_multiple_independent_vowels("অ")
    assert not contains_multiple_independent_vowels("আম")
    assert not contains_multiple_independent_vowels("ক")

    # Rule 4: Contains space or newline (units must not cross whitespace)
    bad_rule4 = ["ক খ", "ক\nখ", "ক\tখ", "ক "]
    for bad in bad_rule4:
        assert contains_space_or_newline(bad), f"Rule 4 failed to fire on {repr(bad)}"
    # Individual whitespace units do not cross whitespace
    assert not contains_space_or_newline(" ")
    assert not contains_space_or_newline("\n")
    assert not contains_space_or_newline("ক")

    # Rule 5: Contains consonant after a vowel sign (second CV core)
    bad_rule5 = ["কাক", "কিখ", "স্ত্রীক", "আকাশ"]
    for bad in bad_rule5:
        assert contains_consonant_after_vowel_sign(bad), f"Rule 5 failed to fire on {repr(bad)}"
    assert not contains_consonant_after_vowel_sign("কা")
    assert not contains_consonant_after_vowel_sign("স্ত্রী")
    assert not contains_consonant_after_vowel_sign("ক্ষ")
    assert not contains_consonant_after_vowel_sign("রক্ত")


def test_wilson_interval_against_known_values() -> None:
    """Wilson score confidence interval must match standard textbook values."""
    # Known case 1: k=50, n=100 (p=0.50) -> [0.4038, 0.5962]
    low1, high1 = wilson_score_interval(50, 100, confidence=0.95)
    assert low1 == pytest.approx(0.4038, abs=1e-3)
    assert high1 == pytest.approx(0.5962, abs=1e-3)

    # Known case 2: k=0, n=100 (p=0.00) -> [0.0000, 0.0370]
    low2, high2 = wilson_score_interval(0, 100, confidence=0.95)
    assert low2 == pytest.approx(0.0000, abs=1e-3)
    assert high2 == pytest.approx(0.0370, abs=1e-3)

    # Known case 3: k=80, n=100 (p=0.80) -> [0.7112, 0.8666]
    low3, high3 = wilson_score_interval(80, 100, confidence=0.95)
    assert low3 == pytest.approx(0.7112, abs=1e-3)
    assert high3 == pytest.approx(0.8666, abs=1e-3)

    # Edge cases
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_review_queue_writer_leaves_owner_verdict_empty(tmp_path: Path) -> None:
    """Review queue TSV writer must emit correct columns and leave owner_verdict empty."""
    sample_disagreements: Counter = Counter({
        (("ক্\u200cস্\u200cফো",), ("ক্\u200c", "স্\u200c", "ফো")): 12,
        (("ন্\u200cস্",), ("ন্\u200c", "স্")): 8,
        (("ল", "\u200c", "্"), ("ল\u200c্",)): 5,
    })
    sample_examples = {
        (("ক্\u200cস্\u200cফো",), ("ক্\u200c", "স্\u200c", "ফো")): ["আলোকস্ফোরণ ঘটিল"],
        (("ন্\u200cস্",), ("ন্\u200c", "স্")): ["রোমান্স প্রিয়"],
        (("ল", "\u200c", "্"), ("ল\u200c্",)): ["বিল্‌কিস বেগম"],
    }

    queue_file = tmp_path / "akshara_review_queue.tsv"
    written_count = write_review_queue(sample_disagreements, sample_examples, queue_file, top_n=100)

    assert written_count == 3
    assert queue_file.is_file()

    content = queue_file.read_text(encoding="utf-8").splitlines()
    assert len(content) == 4  # Header + 3 rows

    expected_header = "pattern_id\tour_split\tbaseline_split\tcount\texample_context\towner_verdict"
    assert content[0] == expected_header

    for line in content[1:]:
        cols = line.split("\t")
        assert len(cols) == 6, f"Expected 6 columns in review queue row, got: {cols}"
        assert cols[5] == "", f"owner_verdict must be empty string, got: {repr(cols[5])}"
        assert cols[0].startswith("P"), f"pattern_id must start with 'P', got: {cols[0]}"
        assert int(cols[3]) > 0, "Count must be positive integer"


def test_gold_set_evaluation_unlabelled_and_rejection(tmp_path: Path) -> None:
    """Gold set evaluator cleanly skips unlabelled TSVs and rejects malformed rows."""
    # 1. Unlabelled TSV: skips cleanly
    unlabelled_tsv = tmp_path / "unlabelled_gold.tsv"
    unlabelled_tsv.write_text("word\taksharas\nরবীন্দ্রনাথ\t\nগীতাঞ্জলি\t\n", encoding="utf-8")

    res_unlab = evaluate_gold_set(unlabelled_tsv)
    assert res_unlab["status"] == "unlabelled"
    assert "skipping gold set evaluation cleanly" in res_unlab["message"].lower()

    # 2. Labelled TSV with valid and malformed rows
    labelled_tsv = tmp_path / "labelled_gold.tsv"
    labelled_content = (
        "word\taksharas\n"
        "রবীন্দ্রনাথ\tর বী ন্দ্র না থ\n"  # Valid & matches segmenter
        "শান্তি\tশা ন্তি\n"            # Valid & matches segmenter
        "ক্ষমা\tখ মা\n"              # Malformed: label without spaces != word -> REJECTED
        "পাহাড়\tপা হা ড়\n"           # Valid
    )
    labelled_tsv.write_text(labelled_content, encoding="utf-8")

    res_lab = evaluate_gold_set(labelled_tsv)
    assert res_lab["status"] == "labelled"
    assert res_lab["total_rows"] == 4
    assert res_lab["labelled_rows"] == 4
    assert res_lab["rejected_rows"] == 1  # 'ক্ষমা' rejected
    assert res_lab["valid_evaluated"] == 3
    assert res_lab["word_accuracy"] == 1.0  # All 3 valid words match
