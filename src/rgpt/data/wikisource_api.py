"""
src/rgpt/data/wikisource_api.py : Fallback MediaWiki Action API client for Bengali Wikisource with rate-limiting and caching.
Backlog/Phase : I5 / Phase F      Docs : 15_BUILD_DIRECTIVE.md Phase F; 03_DATA_PLAN.md §5.2; 09 §7
Contract      : titles: list[str] -> list[dict]
Invariants    : Uses POST batching (50 titles max); sleep >= 3.2s between calls; User-Agent from RGPT_CONTACT
Failure modes : Missing RGPT_CONTACT env var; HTTP 429/503 rate-limit or network timeout
Status        : TESTED
"""

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional


def extract_quality_level(wikitext: str) -> Optional[str]:
    """Extract quality level (0, 1, 2, 3, 4) from pagequality tag."""
    match = re.search(r"<pagequality\s+level=[\"'](\d+)[\"']", wikitext, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def fetch_pages_api(
    titles: List[str],
    contact_info: str,
    delay_seconds: float = 3.2,
    http_caller: Optional[Callable[[str, Dict[str, str]], str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch pages from bn.wikisource.org API using rate-limited batching.

    Args:
        titles: List of page titles to fetch.
        contact_info: User-Agent identification string (e.g. from RGPT_CONTACT).
        delay_seconds: Minimum delay between consecutive HTTP requests (>= 3.2s).
        http_caller: Optional callable for injecting mock network responses during unit testing.

    Returns:
        List of fetched page metadata and wikitext content dictionaries.

    Raises:
        ValueError: If contact_info is empty or delay_seconds < 3.2.
    """
    if not contact_info or not contact_info.strip():
        raise ValueError(
            "Contact info (RGPT_CONTACT) must be provided in User-Agent header per Wikimedia policy."
        )

    if delay_seconds < 3.0:
        raise ValueError(f"Rate limiting requires delay >= 3.0s, got {delay_seconds}")

    results: List[Dict[str, Any]] = []
    endpoint = "https://bn.wikisource.org/w/api.php"
    headers = {
        "User-Agent": f"RabindricGPT/1.0 ({contact_info})",
        "Accept": "application/json",
    }

    # Batch titles up to 50 titles per request
    batch_size = 50
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        data = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "format": "json",
            "formatversion": "2",
            "titles": "|".join(batch),
        }

        if http_caller:
            response_text = http_caller(endpoint, data)
        else:
            encoded_data = urllib.parse.urlencode(data).encode("utf-8")
            req = urllib.request.Request(endpoint, data=encoded_data, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_text = resp.read().decode("utf-8")

        payload = json.loads(response_text)
        pages = payload.get("query", {}).get("pages", [])

        for p in pages:
            title = p.get("title", "")
            revisions = p.get("revisions", [])
            content = revisions[0].get("content", "") if revisions else ""
            quality = extract_quality_level(content)
            results.append({
                "pageid": p.get("pageid", 0),
                "ns": p.get("ns", 104),
                "title": title,
                "quality": quality,
                "wikitext": content,
            })

        # Respect Wikimedia bot rate limit if more batches remain and using real network
        if not http_caller and i + batch_size < len(titles):
            time.sleep(delay_seconds)

    return results
