from datetime import datetime, timezone

import pytest

from mindsim.scrape import (
    HackerNewsClient,
    PricingScraper,
    RedditClient,
    ScrapeBudget,
    ScrapedDocument,
    make_quote_id,
    robots_allows,
)


def test_scraped_document_construction():
    doc = ScrapedDocument(
        source="reddit",
        url="https://reddit.com/r/x/comments/abc",
        retrieved_at=datetime.now(timezone.utc),
        title="title",
        body="body text",
        quote_id=make_quote_id("https://reddit.com/r/x/comments/abc", "body text"),
    )
    assert len(doc.quote_id) == 12


def test_quote_id_deterministic():
    a = make_quote_id("https://u", "snippet")
    b = make_quote_id("https://u", "snippet")
    assert a == b
    assert make_quote_id("https://u", "different") != a


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


def test_reddit_client_returns_empty_in_wave_0():
    assert RedditClient().search("query") == []


def test_hackernews_client_returns_empty_in_wave_0():
    assert HackerNewsClient().search("query") == []


def test_pricing_scraper_returns_none_in_wave_0():
    assert PricingScraper().fetch("https://example.com/pricing") is None


def test_robots_rejects_malformed_url():
    # No scheme or netloc — we should not fetch.
    assert robots_allows("not-a-url") is False
