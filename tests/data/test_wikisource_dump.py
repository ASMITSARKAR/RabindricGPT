"""
tests/data/test_wikisource_dump.py : Tests for Bengali Wikisource XML dump streaming parser.
"""

from pathlib import Path

from src.rgpt.data.wikisource_dump import parse_wikisource_dump

SYNTHETIC_XML = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <page>
    <title>পাতা:গোরা-রবীন্দ্রনাথ ঠাকুর.pdf/1</title>
    <ns>104</ns>
    <revision>
      <text>&lt;noinclude&gt;&lt;pagequality level="0" user="Bot" /&gt;&lt;/noinclude&gt;পাঠ্যহীন প্রচ্ছদ</text>
    </revision>
  </page>
  <page>
    <title>পাতা:গোরা-রবীন্দ্রনাথ ঠাকুর.pdf/2</title>
    <ns>104</ns>
    <revision>
      <text>&lt;noinclude&gt;&lt;pagequality level="1" user="User1" /&gt;&lt;/noinclude&gt;কাঁচা ওসিআর পাঠ্য</text>
    </revision>
  </page>
  <page>
    <title>পাতা:গোরা-রবীন্দ্রনাথ ঠাকুর.pdf/3</title>
    <ns>104</ns>
    <revision>
      <text>&lt;noinclude&gt;&lt;pagequality level="3" user="Proofreader" /&gt;&lt;/noinclude&gt;সংশোধিত প্রথম পরিচ্ছেদ</text>
    </revision>
  </page>
  <page>
    <title>পাতা:গোরা-রবীন্দ্রনাথ ঠাকুর.pdf/4</title>
    <ns>104</ns>
    <revision>
      <text>&lt;noinclude&gt;&lt;pagequality level="4" user="Validator" /&gt;&lt;/noinclude&gt;বৈধকৃত দ্বিতীয় পরিচ্ছেদ</text>
    </revision>
  </page>
  <page>
    <title>পাতা:অন্য-বই.pdf/1</title>
    <ns>104</ns>
    <revision>
      <text>&lt;noinclude&gt;&lt;pagequality level="4" user="Validator" /&gt;&lt;/noinclude&gt;অন্যান্য লেখকের বই</text>
    </revision>
  </page>
</mediawiki>
"""


def test_dump_parser_pagequality_filter(tmp_path: Path) -> None:
    """Parser must strictly select Quality 3 and 4 pages, ignoring Quality 0, 1, and 2."""
    xml_file = tmp_path / "test_dump.xml"
    xml_file.write_text(SYNTHETIC_XML, encoding="utf-8")

    prefixes = ["পাতা:গোরা-রবীন্দ্রনাথ ঠাকুর.pdf/"]
    results = list(parse_wikisource_dump(xml_file, prefixes=prefixes, allowed_qualities=["3", "4"]))

    assert len(results) == 2
    qualities = [r["quality"] for r in results]
    assert "3" in qualities
    assert "4" in qualities
    assert "0" not in qualities
    assert "1" not in qualities


def test_dump_parser_on_synthetic_fixture(tmp_path: Path) -> None:
    """Parser must correctly extract titles and text from synthetic XML fixture."""
    xml_file = tmp_path / "test_dump.xml"
    xml_file.write_text(SYNTHETIC_XML, encoding="utf-8")

    prefixes = ["পাতা:গোরা-রবীন্দ্রনাথ ঠাকুর.pdf/"]
    results = list(parse_wikisource_dump(xml_file, prefixes=prefixes))

    assert results[0]["title"] == "পাতা:গোরা-রবীন্দ্রনাথ ঠাকুর.pdf/3"
    assert "সংশোধিত প্রথম পরিচ্ছেদ" in results[0]["text"]
    assert results[1]["title"] == "পাতা:গোরা-রবীন্দ্রনাথ ঠাকুর.pdf/4"
    assert "বৈধকৃত দ্বিতীয় পরিচ্ছেদ" in results[1]["text"]
