"""
src/rgpt/data/wikitext_clean.py : Stripping ProofreadPage tags, templates, parser functions, and markup.
Backlog/Phase : I5 / Phase F      Docs : 15_BUILD_DIRECTIVE.md Phase F; 03_DATA_PLAN.md §5.2, §10
Contract      : raw_wikitext: str -> (clean_text: str, stats: dict)
Invariants    : No <noinclude> or {{...}} templates remain in cleaned text; poetry newlines preserved
Failure modes : Unclosed HTML or template syntax; empty output
Status        : TESTED
"""

import re
from typing import Any, Dict, Tuple

from src.rgpt.text.clean import bengali_character_ratio, clean_pipeline


def _strip_templates(text: str) -> str:
    """Recursively strip or resolve MediaWiki templates, handling nesting."""
    # Pattern to match innermost template {{...}}
    template_pattern = re.compile(r"\{\{([^{}]+)\}\}")

    def replace_template(match: re.Match) -> str:
        content = match.group(1).strip()
        parts = [p.strip() for p in content.split("|")]
        name = parts[0].lower()

        # Templates that contain text we want to keep (formatting wrappers)
        # e.g., {{কেন্দ্র|টেক্সট}}, {{বড়|টেক্সট}}, {{sc|text}}, {{italics|text}}
        text_keeping_templates = {
            "কেন্দ্র", "ডান", "বাম", "বড়", "ছোট", "অতিবড়", "অতিছোট",
            "bold", "italic", "italics", "sc", "b", "i", "u", "g",
            "block center", "blockcenter", "center", "পদ্য", "কবিতা", "কাব্য",
        }

        if name in text_keeping_templates and len(parts) > 1:
            # Return the last or main text argument
            return parts[-1]
        elif name in ("gap", "স্পেস"):
            return " "
        elif name in ("-", "হরিণ", "line"):
            return "\n"
        # Metadata / headers / references / navigation templates are stripped
        return ""

    # Iterate until no more {{...}} templates remain or safety limit reached
    for _ in range(10):
        new_text, count = template_pattern.subn(replace_template, text)
        if count == 0:
            break
        text = new_text

    # Any remaining unclosed {{ or }} stripped
    text = re.sub(r"\{\{|\}\}", "", text)
    return text


def clean_wikitext(raw_wikitext: str) -> Tuple[str, Dict[str, Any]]:
    """Clean ProofreadPage wikitext: strip headers/footers, templates, and formatting tags.

    Args:
        raw_wikitext: Raw text from MediaWiki XML dump or API.

    Returns:
        Tuple of (clean string, statistics dictionary including char and byte counts).
    """
    raw_chars = len(raw_wikitext)
    raw_bytes = len(raw_wikitext.encode("utf-8"))

    text = raw_wikitext

    # 1. Strip ProofreadPage <noinclude>...</noinclude> header/footer tags
    text = re.sub(r"<noinclude>.*?</noinclude>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?noinclude\s*/?>", "", text, flags=re.IGNORECASE)

    # 2. Strip <pagequality ... /> tags
    text = re.sub(r"<pagequality[^>]*/>", "", text, flags=re.IGNORECASE)

    # 3. Strip HTML tables {| ... |}
    text = re.sub(r"\{\|[\s\S]*?\|\}", "", text)

    # 4. Strip Categories and Files
    text = re.sub(r"\[\[(?:বিষয়|Category|চিত্র|File|Image):.*?\]\]", "", text, flags=re.IGNORECASE)

    # 5. Clean templates recursively
    text = _strip_templates(text)

    # 6. Clean wikilinks [[Target|Anchor]] -> Anchor, [[Page]] -> Page
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)

    # 7. Clean external links [http://example.com Anchor] -> Anchor or strip
    text = re.sub(r"\[https?://[^\s\]]*\s*([^\]]*)\]", r"\1", text)
    text = re.sub(r"https?://[^\s]+", "", text)

    # 8. Strip HTML tags but convert <br> to newline
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)

    # 9. Clean wikitext formatting: bold/italics ''''' -> ''
    text = re.sub(r"'{2,5}", "", text)

    # 10. Clean headings == Heading == -> Heading
    text = re.sub(r"={2,6}\s*(.*?)\s*={2,6}", r"\1\n", text)

    # 11. Clean MediaWiki indentation colons and definition lists
    text = re.sub(r"^\s*:+", "", text, flags=re.MULTILINE)
    text = re.sub(r":{2,}", "", text)

    # 12. Strip any residual unclosed template, link, or tag markup
    text = re.sub(r"\{\{|\}\}", "", text)
    text = re.sub(r"\[\[|\]\]", "", text)
    text = re.sub(r"<[^>]*>", "", text)

    # 11. Run through Bengali clean pipeline (NFC normalization, control char removal,
    # whitespace collapsing while preserving single poetry newlines, ZWJ/ZWNJ preservation)
    clean_text, _passed, _metrics = clean_pipeline(text)

    clean_chars = len(clean_text)
    clean_bytes = len(clean_text.encode("utf-8"))
    ratio = bengali_character_ratio(clean_text) if clean_chars > 0 else 0.0

    stats = {
        "raw_chars": raw_chars,
        "raw_bytes": raw_bytes,
        "clean_chars": clean_chars,
        "clean_bytes": clean_bytes,
        "bengali_ratio": ratio,
        "bytes_per_char": clean_bytes / clean_chars if clean_chars > 0 else 0.0,
    }

    return clean_text, stats
