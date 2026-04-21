"""Wave 0 unit tests for scrape primitives.

Live HTTP behaviour is covered by tests/test_scrape_live.py with mocked
transports.
"""
from datetime import datetime, timezone

import pytest

from mindsim.scrape import (
    ScrapeBudget,
    ScrapedDocument,
    make_quote_id,
    make_source_id,
    robots_allows,
)


def test_scraped_document_construction():
    qid = make_quote_id(
        "https://reddit.com/r/x/comments/abc",
        "body text",
    )
    doc = ScrapedDocument(
        source="reddit",
        url="https://reddit.com/r/x/comments/abc",
        retrieved_at=datetime.now(timezone.utc),
        title="title",
        body="body text",
        quote_id=qid,
    )
    assert doc.quote_id.startswith("voc:")
    assert len(doc.quote_id) == len("voc:") + 12


def test_quote_id_deterministic():
    a = make_quote_id("https://u", "snippet")
    b = make_quote_id("https://u", "snippet")
    assert a == b
    assert make_quote_id("https://u", "different") != a


def test_source_id_distinct_from_quote_id():
    same_inputs = ("https://u", "snippet")
    qid = make_quote_id(*same_inputs)
    sid = make_source_id(*same_inputs)
    assert qid.startswith("voc:")
    assert sid.startswith("src:")
    # Hash payload differs only by prefix → digests should still match.
    assert qid.split(":", 1)[1] == sid.split(":", 1)[1]


def test_scrape_budget_enforcement():
    budget = ScrapeBudget(limit=3)
    assert budget.consume(2) is True
    assert budget.remaining == 1
    assert budget.consume(2) is False  # would exceed
    assert budget.used == 2  # unchanged on rejection


def test_scrape_budget_rejects_negative():
    with pytest.raises(ValueError):
        ScrapeBudget(limit=-1)
    budget = ScrapeBudget(limit=5)
    with pytest.raises(ValueError):
        budget.consume(-1)


def test_robots_rejects_malformed_url():
    # No scheme or netloc — we should not fetch.
    assert robots_allows("not-a-url") is False
