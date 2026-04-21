"""Wave 5: scraper live-behaviour tests with mocked HTTP transports.

Uses httpx.MockTransport so no real network call is made. Each scraper
is exercised end-to-end: budget consumed, rate limiter respected, parser
turns canned JSON / HTML into ScrapedDocument records, errors swallowed.
"""
from __future__ import annotations

import json
import time

import httpx
import pytest

from mindsim.scrape import (
    HackerNewsClient,
    PricingScraper,
    RedditClient,
    ScrapeBudget,
)
from mindsim.scrape import base as scrape_base


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the global rate limiter so tests don't sleep on each other."""
    rl = scrape_base.get_rate_limiter()
    rl._last = 0.0
    yield
    rl._last = 0.0


@pytest.fixture(autouse=True)
def _bypass_robots(monkeypatch):
    """Don't hit reddit.com/robots.txt during unit tests."""
    monkeypatch.setattr(scrape_base, "robots_allows", lambda url, user_agent=None: True)
    # Also patch the symbols already imported into the scraper modules.
    from mindsim.scrape import reddit as red_mod
    from mindsim.scrape import hackernews as hn_mod
    from mindsim.scrape import pricing as price_mod
    monkeypatch.setattr(red_mod, "robots_allows", lambda url: True)
    monkeypatch.setattr(hn_mod, "robots_allows", lambda url: True)
    monkeypatch.setattr(price_mod, "robots_allows", lambda url: True)


def _mock_client(handler):
    """Patched httpx.Client factory that uses MockTransport."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)
    return factory


# ─────────────────────────── Reddit ───────────────────────────


REDDIT_FIXTURE = {
    "data": {
        "children": [
            {
                "data": {
                    "permalink": "/r/programming/comments/abc/title_one/",
                    "title": "AI tool review",
                    "selftext": "I tried this tool for two weeks. The pricing is okay but the UX is rough.",
                    "subreddit": "programming",
                    "score": 42,
                    "num_comments": 7,
                }
            },
            {
                "data": {
                    "permalink": "/r/programming/comments/def/title_two/",
                    "title": "Another tool",
                    "selftext": "",
                    "url": "https://reddit.com/r/programming/comments/def/",
                    "subreddit": "programming",
                    "score": 5,
                    "num_comments": 0,
                }
            },
        ]
    }
}


class TestRedditClient:
    def test_search_parses_listing(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "search.json" in request.url.path
            return httpx.Response(200, json=REDDIT_FIXTURE)
        monkeypatch.setattr(httpx, "Client", _mock_client(handler))

        client = RedditClient(budget=ScrapeBudget(limit=5))
        docs = client.search("ai tool", subreddits=["programming"], limit=10)
        # Second child has empty selftext but non-empty title used as body
        assert len(docs) == 2
        assert docs[0].source == "reddit"
        assert docs[0].quote_id.startswith("voc:")
        assert docs[0].metadata["subreddit"] == "programming"
        assert docs[0].url.startswith("https://www.reddit.com/r/programming/")

    def test_budget_enforced(self, monkeypatch):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(200, json={"data": {"children": []}})
        monkeypatch.setattr(httpx, "Client", _mock_client(handler))

        client = RedditClient(budget=ScrapeBudget(limit=2))
        client.search("q", subreddits=["a", "b", "c", "d"])
        # Only 2 sub-fetches allowed by budget
        assert calls["n"] == 2

    def test_http_error_returns_empty(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="rate limited")
        monkeypatch.setattr(httpx, "Client", _mock_client(handler))

        client = RedditClient(budget=ScrapeBudget(limit=5))
        docs = client.search("q", subreddits=["x"])
        assert docs == []


# ───────────────────────── Hacker News ─────────────────────────


HN_FIXTURE = {
    "hits": [
        {
            "objectID": "111",
            "title": "Show HN: thing",
            "story_text": "We built a thing. It does X better than Y.",
            "url": "https://example.com/post",
            "points": 99,
            "num_comments": 23,
            "author": "alice",
        },
        {
            "objectID": "222",
            "title": "Comment-only story",
            "comment_text": "interesting take on B",
            "url": None,
            "points": 0,
            "num_comments": 0,
            "author": "bob",
        },
    ]
}


class TestHackerNewsClient:
    def test_search_parses_hits(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/search")
            assert request.url.params.get("query") == "ai tool"
            return httpx.Response(200, json=HN_FIXTURE)
        monkeypatch.setattr(httpx, "Client", _mock_client(handler))

        client = HackerNewsClient(budget=ScrapeBudget(limit=5))
        docs = client.search("ai tool", limit=20)
        assert len(docs) == 2
        assert docs[0].source == "hackernews"
        assert docs[0].url == "https://example.com/post"
        assert docs[1].url == "https://news.ycombinator.com/item?id=222"

    def test_consumes_budget(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"hits": []})
        monkeypatch.setattr(httpx, "Client", _mock_client(handler))

        budget = ScrapeBudget(limit=1)
        client = HackerNewsClient(budget=budget)
        client.search("q")
        assert budget.remaining == 0
        # Second call refused
        assert client.search("q") == []

    def test_http_error_returns_empty(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")
        monkeypatch.setattr(httpx, "Client", _mock_client(handler))

        client = HackerNewsClient(budget=ScrapeBudget(limit=5))
        assert client.search("q") == []


# ───────────────────────── Pricing scraper ─────────────────────────


PRICING_HTML = """
<html><head><title>Acme — Pricing</title></head>
<body>
  <h1>Plans</h1>
  <div>Free Tier — get started</div>
  <div>Pro $19/month</div>
  <div>Team $49/mo</div>
</body></html>
"""


class TestPricingScraper:
    def test_fetch_extracts_price(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=PRICING_HTML)
        monkeypatch.setattr(httpx, "Client", _mock_client(handler))

        scraper = PricingScraper(budget=ScrapeBudget(limit=2))
        doc = scraper.fetch("https://acme.com/pricing")
        assert doc is not None
        assert doc.source == "pricing_page"
        # Cheapest non-zero monthly price
        assert doc.metadata["price"] == pytest.approx(19.0)
        assert doc.metadata["price_model"] == "subscription"
        assert doc.metadata["has_free_tier"] is True
        assert doc.title and "Acme" in doc.title

    def test_fetch_freemium_only(self, monkeypatch):
        html = "<html><body>Forever Free. No paid plans.</body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=html)
        monkeypatch.setattr(httpx, "Client", _mock_client(handler))

        scraper = PricingScraper(budget=ScrapeBudget(limit=1))
        doc = scraper.fetch("https://example.com/pricing")
        assert doc is not None
        assert doc.metadata["has_free_tier"] is True
        assert doc.metadata["price"] == 0.0
        assert doc.metadata["price_model"] == "freemium"

    def test_fetch_404_returns_none(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")
        monkeypatch.setattr(httpx, "Client", _mock_client(handler))

        scraper = PricingScraper(budget=ScrapeBudget(limit=1))
        assert scraper.fetch("https://example.com/missing") is None

    def test_fetch_consumes_budget_on_success(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=PRICING_HTML)
        monkeypatch.setattr(httpx, "Client", _mock_client(handler))

        budget = ScrapeBudget(limit=1)
        scraper = PricingScraper(budget=budget)
        scraper.fetch("https://acme.com/pricing")
        assert budget.remaining == 0


# ────────────────────── Rate limiter behaviour ──────────────────────


class TestRateLimiter:
    def test_back_to_back_calls_are_spaced(self, monkeypatch):
        # Force a small interval so the test runs fast.
        rl = scrape_base.get_rate_limiter()
        old_interval = rl.min_interval
        rl.min_interval = 0.05
        try:
            def handler(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json={"data": {"children": []}})
            monkeypatch.setattr(httpx, "Client", _mock_client(handler))

            client = RedditClient(budget=ScrapeBudget(limit=3))
            t0 = time.monotonic()
            client.search("q", subreddits=["a", "b", "c"])
            elapsed = time.monotonic() - t0
            # 3 calls × 0.05s spacing → at least ~0.10s (first is free).
            assert elapsed >= 0.09
        finally:
            rl.min_interval = old_interval
            rl._last = 0.0
