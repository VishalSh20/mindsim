"""Reddit scraper — public JSON endpoints (no auth).

Endpoint: https://www.reddit.com/r/<sub>/search.json?q=<q>&restrict_sr=on&limit=<N>

Reddit's public JSON is rate-limited aggressively. We honour the global
rate limiter (1 req/s), set a User-Agent, respect ScrapeBudget, and fail
open on any HTTP / parse error so the caller (research stage) can
proceed with reduced confidence rather than crash.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

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

REDDIT_BASE = "https://www.reddit.com"
DEFAULT_TIMEOUT_S = 10.0
MAX_BODY_CHARS = 800  # keep VoC quotes compact


class RedditClient:
    def __init__(
        self,
        budget: ScrapeBudget | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        self.budget = budget or ScrapeBudget()
        self.timeout = timeout

    def search(
        self,
        query: str,
        subreddits: Iterable[str] | None = None,
        limit: int = 25,
    ) -> list[ScrapedDocument]:
        """Search reddit. Returns ScrapedDocument list, possibly empty.

        If `subreddits` given, run one query per sub and merge. If not,
        run a global r/all-style search via the search.json endpoint.
        """
        targets = list(subreddits) if subreddits else [None]
        out: list[ScrapedDocument] = []
        for sub in targets:
            if not self.budget.consume(1):
                logger.info("Reddit budget exhausted, stopping search")
                break
            url = self._build_url(query, sub, limit)
            if not robots_allows(url):
                logger.info("robots.txt disallows %s", url)
                continue
            docs = self._fetch_one(url)
            out.extend(docs)
        return out

    def _build_url(self, query: str, sub: str | None, limit: int) -> str:
        encoded = httpx.QueryParams({"q": query, "limit": limit, "restrict_sr": "on"})
        if sub:
            return f"{REDDIT_BASE}/r/{sub}/search.json?{encoded}"
        return f"{REDDIT_BASE}/search.json?{httpx.QueryParams({'q': query, 'limit': limit})}"

    def _fetch_one(self, url: str) -> list[ScrapedDocument]:
        get_rate_limiter().wait()
        try:
            with httpx.Client(timeout=self.timeout, headers={"User-Agent": USER_AGENT}) as client:
                resp = client.get(url)
            if resp.status_code != 200:
                logger.warning("Reddit returned %d for %s", resp.status_code, url)
                return []
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Reddit fetch failed for %s: %s", url, exc)
            return []
        return _parse_listing(payload)


def _parse_listing(payload: dict) -> list[ScrapedDocument]:
    """Parse Reddit `Listing` JSON into ScrapedDocument records."""
    out: list[ScrapedDocument] = []
    children = payload.get("data", {}).get("children", []) or []
    now = datetime.now(timezone.utc)
    for child in children:
        data = child.get("data", {}) or {}
        permalink = data.get("permalink") or ""
        full_url = f"{REDDIT_BASE}{permalink}" if permalink else (data.get("url") or "")
        if not full_url:
            continue
        title = (data.get("title") or "").strip() or None
        body = (data.get("selftext") or data.get("title") or "").strip()
        if not body:
            continue
        body = body[:MAX_BODY_CHARS]
        out.append(
            ScrapedDocument(
                source="reddit",
                url=full_url,
                retrieved_at=now,
                title=title,
                body=body,
                metadata={
                    "subreddit": data.get("subreddit"),
                    "score": int(data.get("score") or 0),
                    "num_comments": int(data.get("num_comments") or 0),
                },
                quote_id=make_quote_id(full_url, body),
            )
        )
    return out
