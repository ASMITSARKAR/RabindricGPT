"""
tests/data/test_wikisource_api.py : Tests for fallback MediaWiki Action API client and rate limiting.
"""

import json

import pytest

from src.rgpt.data.wikisource_api import fetch_pages_api


def test_api_client_enforces_rate_limit() -> None:
    """API client must enforce >= 3.0s interval between requests."""
    with pytest.raises(ValueError, match="Rate limiting requires delay >= 3.0s"):
        fetch_pages_api(["পাতা:গোরা/১"], contact_info="test@example.com", delay_seconds=1.0)


def test_api_client_requires_contact_header() -> None:
    """API client must fail loudly if RGPT_CONTACT header is not supplied."""
    with pytest.raises(ValueError, match="Contact info"):
        fetch_pages_api(["পাতা:গোরা/১"], contact_info="")


def test_api_client_mock_fetch() -> None:
    """Verify batching and parsing logic with mock HTTP caller (no network)."""
    mock_response = json.dumps({
        "query": {
            "pages": [
                {
                    "pageid": 1234,
                    "ns": 104,
                    "title": "পাতা:গোরা/১",
                    "revisions": [{"content": "<pagequality level=\"4\" />প্রথম পাতা"}],
                }
            ]
        }
    })

    def mock_http(url: str, data: dict) -> str:
        return mock_response

    results = fetch_pages_api(
        ["পাতা:গোরা/১"],
        contact_info="test@example.com",
        delay_seconds=3.2,
        http_caller=mock_http,
    )
    assert len(results) == 1
    assert results[0]["title"] == "পাতা:গোরা/১"
    assert results[0]["quality"] == "4"
    assert "প্রথম পাতা" in results[0]["wikitext"]
