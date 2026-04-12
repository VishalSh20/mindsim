"""Stage 1: UNDERSTAND — extract ProductProfile + ResearchPlan.

Takes raw user text → calls LLM → parses into ProductProfile.
The LLM must NOT assign numerical simulation parameters.
"""

from __future__ import annotations

import logging

from mindsim.llm.client import LLMClient
from mindsim.llm.prompts import UNDERSTAND_PROMPT
from mindsim.models.product import (
    CompetitorInfo,
    OptionalQuery,
    ProductProfile,
    ResearchPlan,
    ResearchQuery,
)

logger = logging.getLogger(__name__)


def understand(
    raw_text: str,
    llm: LLMClient,
) -> ProductProfile:
    """Extract structured product information from free text.

    Args:
        raw_text: User's product description in natural language.
        llm: LLM client for making the call.

    Returns:
        ProductProfile with structured extraction and research plan.
    """
    logger.info("Stage 1: Understanding product description...")

    data = llm.complete_json(
        system_prompt=UNDERSTAND_PROMPT,
        user_message=raw_text,
        temperature=0.15,
    )

    # Parse competitors
    competitors = []
    for c in data.get("competitors", []):
        competitors.append(CompetitorInfo(
            name=c.get("name", "Unknown"),
            price=c.get("price"),
            price_model=c.get("price_model"),
        ))

    # Parse research plan
    priority_queries = []
    for q in data.get("research_plan", {}).get("priority_queries", []):
        priority_queries.append(ResearchQuery(
            query=q.get("query", ""),
            goal=q.get("goal", ""),
            confidence_threshold=q.get("confidence_threshold", 0.7),
            fallback_assumption=q.get("fallback_assumption"),
        ))

    optional_queries = []
    for q in data.get("research_plan", {}).get("optional_queries", []):
        optional_queries.append(OptionalQuery(
            query=q.get("query", ""),
            goal=q.get("goal", ""),
            trigger_condition=q.get("trigger_condition", ""),
        ))

    profile = ProductProfile(
        name=data.get("name", "Unknown Product"),
        price=data.get("price"),
        price_model=data.get("price_model"),
        billing_period=data.get("billing_period"),
        competitors=competitors,
        target_audience=data.get("target_audience"),
        target_audience_inferred=data.get("target_audience_inferred", False),
        value_proposition=data.get("value_proposition"),
        category=data.get("category"),
        missing_information=data.get("missing_information", []),
        raw_description=raw_text,
        research_plan=ResearchPlan(
            priority_queries=priority_queries,
            optional_queries=optional_queries,
        ),
    )

    logger.info(f"Understood: {profile.name} (category: {profile.category})")
    return profile
