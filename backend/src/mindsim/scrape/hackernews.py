"""Hacker News scraper via Algolia public API.

Endpoint: https://hn.algolia.com/api/v1/search?query=<q>&tags=story&hitsPerPage=<N>

No auth required. Algolia is generous on rate limits but we honour the
shared 1-req/s rate limiter for politeness. Failures are swallowed —
caller proceeds with empty corpus.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from mindsim.scrape.base import (
    USER_AGENT,
    ScrapeBudget,
    ScrapedDocument,
    get_rate_limiter,
    make_quote_id,
    robots_allows,
)

logger = logging.getLogger(__name__)

HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
DEFAULT_TIMEOUT_S = 10.0
MAX_BODY_CHARS = 800


class HackerNewsClient:
    def __init__(
        self,
        budget: ScrapeBudget | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        self.budget = budget or ScrapeBudget()
        self.timeout = timeout

    def search(self, query: str, limit: int = 25) -> list[ScrapedDocument]:
        if not self.budget.consume(1):
            logger.info("HN budget exhausted")
            return []
        params = {"query": query, "tags": "story", "hitsPerPage": limit}
        url_for_robots = HN_SEARCH_URL
        if not robots_allows(url_for_robots):
            logger.info("robots.txt disallows %s", url_for_robots)
            return []
        get_rate_limiter().wait()
        try:
            with httpx.Client(timeout=self.timeout, headers={"User-Agent": USER_AGENT}) as client:
                resp = client.get(HN_SEARCH_URL, params=params)
            if resp.status_code != 200:
                logger.warning("HN returned %d", resp.status_code)
                return []
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("HN fetch failed: %s", exc)
            return []
        return _parse_hits(payload)


def _parse_hits(payload: dict) -> list[ScrapedDocument]:
    out: list[ScrapedDocument] = []
    hits = payload.get("hits", []) or []
    now = datetime.now(timezone.utc)
    for hit in hits:
        story_id = hit.get("objectID")
        url = hit.get("url") or (
            f"https://news.ycombinator.com/item?id={story_id}" if story_id else None
        )
        if not url:
            continue
        title = (hit.get("title") or "").strip() or None
        body = (hit.get("story_text") or hit.get("comment_text") or hit.get("title") or "").strip()
        if not body:
            continue
        body = body[:MAX_BODY_CHARS]
        out.append(
            ScrapedDocument(
                source="hackernews",
                url=url,
                retrieved_at=now,
                title=title,
                body=body,
                metadata={
                    "points": int(hit.get("points") or 0),
                    "num_comments": int(hit.get("num_comments") or 0),
                    "author": hit.get("author"),
                },
                quote_id=make_quote_id(url, body),
            )
        )
    return out
