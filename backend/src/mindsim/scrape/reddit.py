"""Reddit scraper.

Wave 0 is a skeleton. Wave 5 wires the real call against
`https://www.reddit.com/r/<sub>/search.json?q=<query>` (public, no auth).
"""
from __future__ import annotations

import logging
from typing import Iterable

from mindsim.scrape.base import ScrapeBudget, ScrapedDocument

logger = logging.getLogger(__name__)


class RedditClient:
    def __init__(self, budget: ScrapeBudget | None = None):
        self.budget = budget or ScrapeBudget()

    def search(
        self,
        query: str,
        subreddits: Iterable[str] | None = None,
        limit: int = 25,
    ) -> list[ScrapedDocument]:
        logger.debug(
            "RedditClient.search(%r, subreddits=%s, limit=%d) [Wave 0 stub]",
            query,
            list(subreddits) if subreddits else None,
            limit,
        )
        return []
