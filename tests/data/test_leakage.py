"""
tests/data/test_leakage.py : Tests for cross-corpus contamination detection and planted passage removal.
"""

from src.rgpt.data.leakage import CrossCorpusLeakageDetector


def test_leakage_purge_removes_planted_tagore_passage() -> None:
    """Pretraining documents containing planted Tagore passages must be purged."""
    target_texts = [
        "আমার মাথা নত করে দাও হে তোমার চরণধূলার তলে। সকল অহংকার হে আমার ডুবাও চোখের জলে।"
    ]
    detector = CrossCorpusLeakageDetector(
        target_texts=target_texts, ngram_size=5, window_size=30, threshold_distinct_ngrams=3
    )

    # Document containing planted passage
    leaking_doc = {
        "id": "leak_1",
        "text": "একটি বিশেষ গান হলো আমার মাথা নত করে দাও হে তোমার চরণধূলার তলে। সকল অহংকার এবং আরও কিছু কথা।",
    }
    # Completely distinct document
    clean_doc = {
        "id": "clean_1",
        "text": "কম্পিউটার বিজ্ঞান হলো পরিগণনা এবং তথ্য প্রক্রিয়াকরণ সংক্রান্ত একটি আধুনিক জ্ঞানশাখা।",
    }

    assert detector.contains_leakage(leaking_doc["text"])
    assert not detector.contains_leakage(clean_doc["text"])

    purged = list(detector.purge_leakage([leaking_doc, clean_doc]))
    assert len(purged) == 1
    assert purged[0]["id"] == "clean_1"


def test_planted_twenty_test_13grams_detected_and_purged() -> None:
    """Item 6: Plant 20 test 13-grams in pretraining docs and prove 100% (20/20) are caught and purged."""
    from pathlib import Path

    from src.rgpt.data.leakage import tokenize_bengali_words

    test_path = Path("data/cleaned/tagore/test.txt")
    if not test_path.exists():
        return
    test_text = test_path.read_text(encoding="utf-8")
    words = tokenize_bengali_words(test_text)

    ngrams_13 = []
    step_size = max(1, len(words) // 25)
    for i in range(20):
        start = i * step_size
        gram = " ".join(words[start : start + 13])
        ngrams_13.append(gram)

    detector = CrossCorpusLeakageDetector(
        target_texts=[test_text], ngram_size=5, window_size=30, threshold_distinct_ngrams=3
    )
    background_text = "বাংলা ভাষা ও সাহিত্যের ধারাবাহিক ইতিহাস অত্যন্ত সমৃদ্ধ ও প্রাচীন।"

    planted_docs = []
    for i, gram in enumerate(ngrams_13):
        planted_docs.append({
            "id": f"planted_{i}",
            "text": f"{background_text} {gram} {background_text}",
        })

    caught = [detector.contains_leakage(d["text"]) for d in planted_docs]
    assert sum(caught) == 20, f"Only caught {sum(caught)}/20 planted 13-grams!"
    assert not detector.contains_leakage(background_text), "False positive on background!"

    purged = list(detector.purge_leakage(planted_docs))
    assert len(purged) == 0, f"Expected 0 docs after purge, got {len(purged)}"

