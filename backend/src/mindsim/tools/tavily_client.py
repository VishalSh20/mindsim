"""Tavily search wrapper — budget-tracked research API client."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class TavilyClient:
    """Wrapper around tavily-python with credit tracking."""

    def __init__(self, budget: int | None = None):
        self.api_key = os.getenv("TAVILY_API_KEY", "")
        self.budget = budget or int(os.getenv("MINDSIM_RESEARCH_BUDGET", "10"))
        self.credits_used = 0
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "TAVILY_API_KEY not set. Set it in .env or use --skip-research."
                )
            from tavily import TavilyClient as _Tavily
            self._client = _Tavily(api_key=self.api_key)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.budget - self.credits_used)

    def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
    ) -> dict:
        """Run a Tavily search.

        Args:
            query: Search query.
            search_depth: "basic" (1 credit) or "advanced" (2 credits).
            max_results: Number of results to return.

        Returns:
            Dict with 'results' list, each having 'title', 'url', 'content', 'score'.
        """
        cost = 1 if search_depth == "basic" else 2
        if self.credits_used + cost > self.budget:
            logger.warning(f"Budget exhausted ({self.credits_used}/{self.budget})")
            return {"results": [], "budget_exhausted": True}

        self._ensure_client()

        try:
            response = self._client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
            )
            self.credits_used += cost
            logger.info(
                f"Tavily search: '{query}' ({self.credits_used}/{self.budget} credits)"
            )
            return response
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return {"results": [], "error": str(e)}

    def extract_pricing(self, results: dict) -> list[dict]:
        """Extract pricing data from search results.

        Returns list of {name, price, source_url} dicts.
        """
        pricing = []
        for result in results.get("results", []):
            content = result.get("content", "").lower()
            # Heuristic: look for price patterns
            import re
            price_patterns = re.findall(
                r'\$(\d+(?:\.\d{2})?)\s*(?:/\s*(?:mo|month|yr|year))?',
                content,
            )
            if price_patterns:
                pricing.append({
                    "source": result.get("title", ""),
                    "url": result.get("url", ""),
                    "prices_found": [float(p) for p in price_patterns],
                    "content_snippet": content[:200],
                })
        return pricing
