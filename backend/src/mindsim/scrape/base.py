"""Shared scaffolding for scraping clients.

Wave 0 provides only the shared ScrapedDocument shape, a single global
rate limiter, robots.txt check, and budget tracking. Actual HTTP fetches
are wired in Wave 5 (Reddit/HN JSON APIs) and don't live here yet.

Contract: every scraper returns structured ScrapedDocument records.
Raw HTML must never leave this module.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from datetime import datetime
from typing import Any, Literal
from urllib import robotparser
from urllib.parse import urlparse

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

USER_AGENT = "mindsim-research/0.2"


class ScrapedDocument(BaseModel):
    source: Literal["reddit", "hackernews", "pricing_page"]
    url: str
    retrieved_at: datetime
    title: str | None = None
    body: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    quote_id: str


def make_quote_id(url: str, snippet: str) -> str:
    """Deterministic 12-char id for referencing a quote across the pipeline."""
    digest = hashlib.sha256(f"{url}|{snippet}".encode("utf-8")).hexdigest()
    return digest[:12]


class RateLimiter:
    """Minimum-interval rate limiter, thread-safe."""

    def __init__(self, min_interval_seconds: float = 1.0):
        self.min_interval = min_interval_seconds
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()


_global_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _global_rate_limiter


def robots_allows(url: str, user_agent: str = USER_AGENT) -> bool:
    """Return True if robots.txt permits the URL. Fails open on network/parse errors."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception as exc:
        logger.debug("robots.txt check failed for %s: %s (failing open)", url, exc)
        return True


class ScrapeBudget:
    """Per-run budget. Rejects over-limit consume() atomically."""

    def __init__(self, limit: int = 25):
        if limit < 0:
            raise ValueError("limit must be non-negative")
        self.limit = limit
        self.used = 0

    def consume(self, n: int = 1) -> bool:
        if n < 0:
            raise ValueError("n must be non-negative")
        if self.used + n > self.limit:
            return False
        self.used += n
        return True

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)
