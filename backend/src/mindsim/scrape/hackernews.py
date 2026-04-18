"""Hacker News scraper via Algolia.

Wave 0 is a skeleton. Wave 5 wires the real call against
`https://hn.algolia.com/api/v1/search?query=<q>`.
"""
from __future__ import annotations

import logging

from mindsim.scrape.base import ScrapeBudget, ScrapedDocument

logger = logging.getLogger(__name__)


class HackerNewsClient:
    def __init__(self, budget: ScrapeBudget | None = None):
        self.budget = budget or ScrapeBudget()

    def search(self, query: str, limit: int = 25) -> list[ScrapedDocument]:
        logger.debug(
            "HackerNewsClient.search(%r, limit=%d) [Wave 0 stub]", query, limit
        )
        return []
