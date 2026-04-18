"""Competitor pricing page scraper.

Wave 0 is a skeleton. Wave 5 wires httpx + BeautifulSoup with per-domain
extractors keyed on known pricing URL patterns surfaced by Tavily.
"""
from __future__ import annotations

import logging

from mindsim.scrape.base import ScrapeBudget, ScrapedDocument

logger = logging.getLogger(__name__)


class PricingScraper:
    def __init__(self, budget: ScrapeBudget | None = None):
        self.budget = budget or ScrapeBudget()

    def fetch(self, url: str) -> ScrapedDocument | None:
        logger.debug("PricingScraper.fetch(%r) [Wave 0 stub]", url)
        return None
