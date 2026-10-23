"""
tests/data/test_wikitext_clean.py : Tests for ProofreadPage header/footer and template stripping.
"""

from src.rgpt.data.wikitext_clean import clean_wikitext


def test_wikitext_stripping_removes_templates() -> None:
    """All {{...}} templates and <noinclude> tags must be completely removed."""
    raw = (
        "<noinclude><pagequality level=\"4\" user=\"TestUser\" />\n"
        "{{header|title=গীতাঞ্জলি|author=রবীন্দ্রনাথ}}\n"
        "</noinclude>\n"
        "আমার মাথা নত করে দাও হে তোমার চরণধূলার তলে।\n"
        "{{gap}}সকল অহংকার হে আমার ডুবাও চোখের জলে।\n"
        "<noinclude>\n"
        "{{footer|next=পাতা ২}}\n"
        "</noinclude>"
    )
    clean, stats = clean_wikitext(raw)
    assert "<noinclude>" not in clean
    assert "</noinclude>" not in clean
    assert "pagequality" not in clean
    assert "{{header" not in clean
    assert "{{footer" not in clean
    assert "{{gap}}" not in clean
    assert "আমার মাথা নত করে দাও" in clean
    assert "চরণধূলার তলে" in clean
    assert stats["clean_chars"] > 0
    assert stats["bengali_ratio"] > 0.85


def test_wikitext_stripping_preserves_poetry_newlines() -> None:
    """Poetry line breaks within stanzas must be preserved."""
    raw = (
        "<noinclude><pagequality level=\"3\" user=\"Editor\" /></noinclude>\n"
        "যেতে নাহি দিব। হায়,\n"
        "তবু যেতে দিতে হয়,\n"
        "তবু চলে যায়।\n"
    )
    clean, stats = clean_wikitext(raw)
    lines = clean.split("\n")
    # Verify that there are 3 distinct lines preserved
    assert len(lines) == 3
    assert lines[0] == "যেতে নাহি দিব। হায়,"
    assert lines[1] == "তবু যেতে দিতে হয়,"
    assert lines[2] == "তবু চলে যায়।"
