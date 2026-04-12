"""Stage 2: RESEARCH — adaptive Tavily + Google Trends.

Executes the research plan from the understand stage:
  Phase 1: Priority queries (always run)
  Phase 2: Google Trends (always, free)
  Phase 3: Optional queries (conditional)

Confidence evaluation is heuristic-based — NO LLM calls.
Skippable with --skip-research.
"""

from __future__ import annotations

import logging
import re

from mindsim.models.market import MarketContext, VerifiedCompetitor
from mindsim.models.product import ProductProfile
from mindsim.tools.tavily_client import TavilyClient
from mindsim.tools.trends import get_category_trends

logger = logging.getLogger(__name__)


def research(
    profile: ProductProfile,
    skip: bool = False,
    budget: int | None = None,
) -> MarketContext:
    """Execute the adaptive research loop.

    Args:
        profile: ProductProfile with ResearchPlan from understand stage.
        skip: If True, return empty MarketContext (--skip-research).
        budget: Max Tavily credits to spend.

    Returns:
        MarketContext with verified data and confidence flags.
    """
    if skip:
        logger.info("Research skipped (--skip-research)")
        return MarketContext.empty()

    logger.info("Stage 2: Researching market context...")

    tavily = TavilyClient(budget=budget)
    context = MarketContext()
    low_confidence = []

    # Phase 1: Priority queries
    for query in profile.research_plan.priority_queries:
        logger.info(f"Research query: {query.query}")
        result = tavily.search(query.query, search_depth="basic")

        if result.get("budget_exhausted"):
            low_confidence.append(f"Budget exhausted before: {query.goal}")
            if query.fallback_assumption:
                low_confidence.append(f"Fallback: {query.fallback_assumption}")
            continue

        confidence = _evaluate_confidence(result, query.goal)

        if confidence >= query.confidence_threshold:
            _integrate_result(context, result, query.goal)
        elif tavily.budget_remaining >= 2:
            # Refine with advanced search
            logger.info(f"Low confidence ({confidence:.2f}), refining...")
            refined_query = _refine_query(query.query)
            result2 = tavily.search(refined_query, search_depth="advanced")
            confidence2 = _evaluate_confidence(result2, query.goal)

            if confidence2 >= query.confidence_threshold:
                _integrate_result(context, result2, query.goal)
            else:
                low_confidence.append(
                    f"Low confidence for: {query.goal}"
                    + (f" (fallback: {query.fallback_assumption})" if query.fallback_assumption else "")
                )
        else:
            low_confidence.append(f"Could not verify: {query.goal}")
            if query.fallback_assumption:
                low_confidence.append(f"Using fallback: {query.fallback_assumption}")

    # Phase 2: Google Trends (always, free)
    if profile.category:
        keywords = [profile.category]
        if profile.name:
            keywords.append(profile.name)
        trends = get_category_trends(keywords)
        context.category_growth = trends.get("growth_rate")
        context.category_maturity = trends.get("maturity")
        logger.info(
            f"Trends: {context.category_maturity} "
            f"(growth: {context.category_growth:.2f})"
        )

    # Phase 3: Optional queries (conditional)
    for opt_query in profile.research_plan.optional_queries:
        if _should_trigger(opt_query.trigger_condition, context):
            if tavily.budget_remaining >= 1:
                logger.info(f"Optional query triggered: {opt_query.query}")
                result = tavily.search(opt_query.query)
                _integrate_result(context, result, opt_query.goal)

    context.low_confidence_flags = low_confidence
    context.research_cost = tavily.credits_used

    logger.info(
        f"Research complete: {tavily.credits_used} credits, "
        f"{len(context.verified_competitors)} verified competitors, "
        f"{len(low_confidence)} low-confidence flags"
    )

    return context


def _evaluate_confidence(result: dict, goal: str) -> float:
    """Heuristic confidence evaluation — NO LLM call.

    Evaluates:
    1. Number of results
    2. Source consistency (do results agree?)
    3. Source authority (official pages score higher)
    4. Keyword match with goal
    """
    results = result.get("results", [])
    if not results:
        return 0.0

    score = 0.0

    # Factor 1: Result count (more = better, up to 0.3)
    score += min(len(results) / 5.0, 1.0) * 0.3

    # Factor 2: Source authority (0.3)
    authority_domains = [
        "pricing", "official", ".com", "blog", "techcrunch",
        "crunchbase", "producthunt", "g2.com",
    ]
    authority_score = 0.0
    for r in results:
        url = r.get("url", "").lower()
        if any(d in url for d in authority_domains):
            authority_score += 1
    score += min(authority_score / len(results), 1.0) * 0.3

    # Factor 3: Keyword match with goal (0.2)
    goal_words = set(goal.lower().split())
    content_words = set()
    for r in results[:3]:
        content_words.update(r.get("content", "").lower().split()[:50])
    overlap = len(goal_words & content_words) / max(len(goal_words), 1)
    score += overlap * 0.2

    # Factor 4: Score consistency across results (0.2)
    if len(results) >= 2:
        scores = [r.get("score", 0) for r in results]
        if max(scores) > 0:
            consistency = min(scores) / max(scores)
            score += consistency * 0.2

    return min(score, 1.0)


def _integrate_result(context: MarketContext, result: dict, goal: str):
    """Integrate search result into MarketContext."""
    for r in result.get("results", []):
        content = r.get("content", "")
        title = r.get("title", "")
        url = r.get("url", "")

        # Try to extract competitor pricing info
        prices = re.findall(
            r'\$(\d+(?:\.\d{2})?)\s*(?:/\s*(?:mo|month|yr|year))?',
            content,
        )

        # Check for free tier mentions
        has_free = bool(re.search(
            r'\b(free tier|free plan|freemium|free version)\b',
            content, re.IGNORECASE,
        ))

        if prices:
            comp = VerifiedCompetitor(
                name=title[:60],
                confirmed_price=float(prices[0]),
                has_free_tier=has_free,
                source_url=url,
            )
            context.verified_competitors.append(comp)

        # Try to extract penetration data
        penetration_match = re.search(
            r'(\d+(?:\.\d+)?)\s*%\s*(?:of|penetration|adoption|market share)',
            content, re.IGNORECASE,
        )
        if penetration_match and context.category_penetration is None:
            context.category_penetration = float(penetration_match.group(1)) / 100.0


def _refine_query(query: str) -> str:
    """Refine a search query for better results (rule-based, no LLM)."""
    refinements = [
        " pricing",
        " market share",
        " competitors",
    ]
    # Add the most relevant refinement
    for r in refinements:
        if r.strip() not in query.lower():
            return query + r
    return query + " detailed"


def _should_trigger(condition: str, context: MarketContext) -> bool:
    """Check if an optional query's trigger condition is met."""
    condition_lower = condition.lower()

    if "ambiguous" in condition_lower and "pricing" in condition_lower:
        # Trigger if we don't have clear pricing data
        return len(context.verified_competitors) < 2

    if "no competitors" in condition_lower:
        return len(context.verified_competitors) == 0

    if "low confidence" in condition_lower:
        return len(context.low_confidence_flags) > 0

    # Default: don't trigger
    return False
