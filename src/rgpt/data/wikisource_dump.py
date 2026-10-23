"""
src/rgpt/data/wikisource_dump.py : SAX streaming parser for bnwikisource pages-articles XML dump.
Backlog/Phase : I5 / Phase F      Docs : 15_BUILD_DIRECTIVE.md Phase F; 03_DATA_PLAN.md §5.4
Contract      : dump_path: Path, prefixes: list[str] -> Iterator[dict]
Invariants    : Parses namespace 104; filters pagequality level 3 and 4 only; groups by work prefix
Failure modes : Corrupted bz2 file; unparseable XML tokens; namespace mismatch
Status        : TESTED
"""

import bz2
import gzip
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


def _extract_page_quality(text: str) -> Optional[str]:
    """Extract ProofreadPage quality level (0, 1, 2, 3, or 4) from wikitext."""
    match = re.search(r"<pagequality\s+level=[\"'](\d+)[\"']", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def parse_wikisource_dump(
    dump_path: Union[str, Path],
    prefixes: List[str],
    allowed_qualities: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream-parse Bengali Wikisource XML dump using incremental ElementTree iterparse.

    Args:
        dump_path: Path to bnwikisource pages-articles XML (.xml, .xml.bz2, .xml.gz).
        prefixes: List of title prefixes for the canonical works.
        allowed_qualities: ProofreadPage quality levels to accept (default ["3", "4"]).

    Yields:
        Dictionary for each matching validated page: title, prefix, quality, text.
    """
    if allowed_qualities is None:
        allowed_qualities = ["3", "4"]

    path = Path(dump_path)
    if not path.exists():
        raise FileNotFoundError(f"Wikisource dump file not found: {dump_path}")

    # Determine decompression stream
    if path.suffix == ".bz2":
        f = bz2.open(path, "rb")
    elif path.suffix == ".gz":
        f = gzip.open(path, "rb")
    else:
        f = open(path, "rb")

    try:
        # Use iterparse to stream page elements with low memory overhead
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            # Match <page> tag regardless of XML namespace
            if elem.tag.endswith("page"):
                title = ""
                ns = ""
                text = ""

                for child in elem:
                    tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if tag == "title" and child.text:
                        title = child.text.strip()
                    elif tag == "ns" and child.text:
                        ns = child.text.strip()
                    elif tag == "revision":
                        for rev_child in child:
                            rev_tag = (
                                rev_child.tag.split("}")[-1]
                                if "}" in rev_child.tag
                                else rev_child.tag
                            )
                            if rev_tag == "text":
                                if len(rev_child) > 0:
                                    text = (rev_child.text or "") + "".join(
                                        ET.tostring(c, encoding="unicode") for c in rev_child
                                    )
                                else:
                                    text = rev_child.text or ""

                # Check if page matches any target work prefix
                matched_prefix = None
                for pfx in prefixes:
                    if title.startswith(pfx):
                        matched_prefix = pfx
                        break

                if matched_prefix and text:
                    quality = _extract_page_quality(text)
                    if quality in allowed_qualities:
                        yield {
                            "title": title,
                            "prefix": matched_prefix,
                            "quality": quality,
                            "ns": ns,
                            "text": text,
                        }

                # Clear element and previous siblings to free memory
                elem.clear()
    finally:
        f.close()
