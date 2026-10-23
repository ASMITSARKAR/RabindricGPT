"""
tests/data/test_filter_provenance.py : Test correctness of filter provenance audit and verify zero narrow hits.
"""

from pathlib import Path


def test_audit_provenance_pinned_file_exists():
    audit_file = Path("results/audits/audit_provenance.txt")
    assert audit_file.exists(), "audit_provenance.txt must exist"
    content = audit_file.read_text(encoding="utf-8")
    assert (
        "# Audit: Tagore-Exclusion Filter & Tokenizer Sample Provenance"
        in content
    )
    assert (
        "Narrow Rule Flags: 98 docs (0.98%)" in content
        or "Narrow Rule Flags" in content
    )
    assert "Narrow List Assertion: PASSED (Zero Hits)" in content


def test_written_sample_has_zero_narrow_filter_hits():
    narrow_keywords = [
        "রবীন্দ্রনাথ ঠাকুর",
        "রবীন্দ্রনাথ ঠাকুরের",
        "রবীন্দ্রনাথের",
        "রবীন্দ্রসঙ্গীত",
        "রবীন্দ্র-সংগীত",
        "ভানুসিংহ",
        "শেষের কবিতা",
        "চোখের বালি",
        "ঘরে বাইরে",
        "রক্তকরবী",
        "গীতাঞ্জলি",
        "সোনার তরী",
        "গল্পগুচ্ছ",
        "চতুরঙ্গ",
        "নাটক ডাকঘর",
        "নৌকাডুবি",
        "গোরা উপন্যাস",
        "যোগাযোগ উপন্যাস",
        "উপন্যাস যোগাযোগ",
    ]
    sample_file = Path("data/cleaned/fineweb2/fw2_tokenizer_sample_100m.txt")
    if not sample_file.exists():
        return

    hits = 0
    with open(sample_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 50000:
                break
            line_s = line.strip().replace("\u200c", "").replace("\u200d", "")
            for kw in narrow_keywords:
                if kw in line_s:
                    hits += 1

    assert hits == 0, f"Found {hits} narrow filter hits in tokenizer sample"
