"""Stage 1: UNDERSTAND — extract ProductProfile + ResearchPlan.

Takes raw user text → calls LLM → parses into ProductProfile.
The LLM must NOT assign numerical simulation parameters.

After extraction, a validation LLM call cross-checks the extracted data
against the raw input to catch errors like price confusion.
"""

from __future__ import annotations

import json
import logging

from mindsim.llm.client import LLMClient
from mindsim.llm.prompts import UNDERSTAND_PROMPT, VALIDATION_PROMPT
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
        Includes validation metadata if corrections were applied.
    """
    logger.info("Stage 1: Understanding product description...")

    data = llm.complete_json(
        system_prompt=UNDERSTAND_PROMPT,
        user_message=raw_text,
        temperature=0.15,
    )

    # Validate extraction against raw input
    validation = _validate_extraction(raw_text, data, llm)
    if validation and validation.get("has_critical_errors"):
        corrections = validation.get("corrected_fields", {})
        for field, value in corrections.items():
            if field in data:
                logger.warning(
                    f"Validation corrected {field}: {data[field]} → {value}"
                )
                data[field] = value

    # Parse competitors (handle both dict entries and bare strings from validation)
    competitors = []
    for c in data.get("competitors", []):
        if isinstance(c, dict):
            competitors.append(CompetitorInfo(
                name=c.get("name", "Unknown"),
                price=c.get("price"),
                price_model=c.get("price_model"),
            ))
        elif isinstance(c, str):
            competitors.append(CompetitorInfo(name=c))

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

    # Attach validation metadata for diagnostic dump
    profile._validation = validation

    logger.info(f"Understood: {profile.name} (category: {profile.category})")
    return profile


def _validate_extraction(
    raw_text: str,
    extracted: dict,
    llm: LLMClient,
) -> dict | None:
    """Run a validation LLM call to catch extraction errors.

    Cross-checks extracted data against the raw user input.
    Returns the validation result dict, or None if validation fails.
    """
    # Build a clean summary of what was extracted (no research plan noise)
    extracted_summary = {
        k: v for k, v in extracted.items()
        if k not in ("research_plan", "missing_information")
    }

    user_message = (
        f"ORIGINAL USER INPUT:\n{raw_text}\n\n"
        f"EXTRACTED DATA:\n{json.dumps(extracted_summary, indent=2, default=str)}"
    )

    try:
        result = llm.complete_json(
            system_prompt=VALIDATION_PROMPT,
            user_message=user_message,
            temperature=0.1,
        )
        if result.get("has_critical_errors"):
            logger.warning(
                f"Validation found critical errors: "
                f"{[v.get('reason', '') for v in result.get('validations', []) if v.get('status') == 'WRONG']}"
            )
        return result
    except Exception as e:
        logger.warning(f"Validation call failed (non-fatal): {e}")
        return None
