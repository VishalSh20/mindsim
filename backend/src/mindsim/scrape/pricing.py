"""Competitor pricing-page scraper.

Per-domain extractor registry handles known patterns; everything else
falls back to a heuristic regex sweep over visible text.

Output: ScrapedDocument with extracted price hints in `metadata`:
  - price: float | None  (USD/month equivalent, best-effort)
  - has_free_tier: bool
  - price_model: str | None  ("subscription" | "one_time" | "freemium")

This is intentionally best-effort — pricing pages are notoriously
inconsistent. Caller (research/synthesizer) cross-references with
Tavily-derived prices and uses the median.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from mindsim.scrape.base import (
    USER_AGENT,
    ScrapeBudget,
    ScrapedDocument,
    get_rate_limiter,
    make_quote_id,
    robots_allows,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10.0
MAX_BODY_CHARS = 1500


# Per-domain extractors. Each maps domain → callable(soup) → (price, model, has_free).
# Stays small for v2-middle; expand opportunistically.
_EXTRACTORS: dict[str, "callable"] = {}


def _register(domain: str):
    def deco(fn):
        _EXTRACTORS[domain] = fn
        return fn
    return deco


@_register("openai.com")
def _openai_extractor(soup: BeautifulSoup) -> tuple[float | None, str | None, bool]:
    text = soup.get_text(" ", strip=True)
    has_free = bool(re.search(r"\bfree\b", text, re.IGNORECASE))
    m = re.search(r"\$(\d+(?:\.\d{1,2})?)\s*/\s*(?:month|mo)", text, re.IGNORECASE)
    return (float(m.group(1)) if m else None, "subscription", has_free)


def _heuristic_extractor(soup: BeautifulSoup) -> tuple[float | None, str | None, bool]:
    """Fallback: visible-text regex sweep."""
    text = soup.get_text(" ", strip=True)
    has_free = bool(
        re.search(r"\b(free tier|free plan|freemium|free version|forever free)\b",
                  text, re.IGNORECASE)
    )
    monthly = re.findall(r"\$(\d+(?:\.\d{1,2})?)\s*/\s*(?:mo|month)", text, re.IGNORECASE)
    annual = re.findall(r"\$(\d+(?:\.\d{1,2})?)\s*/\s*(?:yr|year|annually)", text, re.IGNORECASE)
    one_time = re.findall(r"\$(\d+(?:\.\d{1,2})?)\s*one[\s-]*time", text, re.IGNORECASE)

    if monthly:
        # Cheapest non-zero monthly price (skip zeros from free-tier rows).
        prices = sorted(float(p) for p in monthly if float(p) > 0)
        return (prices[0] if prices else None, "subscription", has_free)
    if annual:
        prices = sorted(float(p) / 12.0 for p in annual if float(p) > 0)
        return (prices[0] if prices else None, "subscription", has_free)
    if one_time:
        prices = sorted(float(p) for p in one_time if float(p) > 0)
        return (prices[0] if prices else None, "one_time", has_free)
    if has_free:
        return (0.0, "freemium", True)
    return (None, None, has_free)


class PricingScraper:
    def __init__(
        self,
        budget: ScrapeBudget | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ):
        self.budget = budget or ScrapeBudget()
        self.timeout = timeout

    def fetch(self, url: str) -> ScrapedDocument | None:
        if not self.budget.consume(1):
            logger.info("pricing budget exhausted")
            return None
        if not robots_allows(url):
            logger.info("robots.txt disallows %s", url)
            return None
        get_rate_limiter().wait()
        try:
            with httpx.Client(
                timeout=self.timeout,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            ) as client:
                resp = client.get(url)
            if resp.status_code != 200:
                logger.warning("pricing fetch %d for %s", resp.status_code, url)
                return None
            html = resp.text
        except httpx.HTTPError as exc:
            logger.warning("pricing fetch failed for %s: %s", url, exc)
            return None

        soup = BeautifulSoup(html, "html.parser")
        domain = urlparse(url).netloc.lower().lstrip("www.")
        extractor = _EXTRACTORS.get(domain, _heuristic_extractor)
        try:
            price, model, has_free = extractor(soup)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pricing extractor failed for %s: %s", domain, exc)
            price, model, has_free = _heuristic_extractor(soup)

        body = soup.get_text(" ", strip=True)[:MAX_BODY_CHARS]
        if not body:
            return None
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        return ScrapedDocument(
            source="pricing_page",
            url=url,
            retrieved_at=datetime.now(timezone.utc),
            title=title,
            body=body,
            metadata={
                "price": price,
                "price_model": model,
                "has_free_tier": has_free,
                "domain": domain,
            },
            quote_id=make_quote_id(url, body),
        )
