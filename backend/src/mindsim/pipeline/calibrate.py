"""Stage 3: CALIBRATE — assign ALL numerical simulation parameters.

Takes ProductProfile + MarketContext → calls LLM → parses into SimulationConfig.
This is the ONLY place numerical parameters are assigned.
"""

from __future__ import annotations

import json
import logging

from mindsim.llm.client import LLMClient
from mindsim.llm.prompts import CALIBRATE_PROMPT
from mindsim.models.config import (
    Assumption,
    AwarenessByArchetype,
    CalibratedParam,
    PopulationConfig,
    ReferencePriceComponent,
    ReferencePriceParam,
    SimulationConfig,
    SimulationParams,
)
from mindsim.models.market import MarketContext
from mindsim.models.product import ProductProfile

logger = logging.getLogger(__name__)


def calibrate(
    profile: ProductProfile,
    market: MarketContext,
    llm: LLMClient,
) -> SimulationConfig:
    """Assign all simulation parameters via LLM calibration.

    Args:
        profile: Product profile from understand stage.
        market: Market context from research stage.
        llm: LLM client.

    Returns:
        SimulationConfig with all numerical parameters calibrated.
    """
    logger.info("Stage 3: Calibrating simulation parameters...")

    # Build context message with product + market data
    context = _build_calibration_context(profile, market)

    data = llm.complete_json(
        system_prompt=CALIBRATE_PROMPT,
        user_message=context,
        temperature=0.15,
    )

    config = _parse_calibration_response(data, profile)

    logger.info(
        f"Calibrated: price=${config.simulation_params.price:.2f}, "
        f"ref_price=${config.simulation_params.reference_price.value:.2f}, "
        f"β={config.simulation_params.present_bias_beta.value:.2f}"
    )
    return config


def _build_calibration_context(
    profile: ProductProfile,
    market: MarketContext,
) -> str:
    """Build the user message for calibration."""
    parts = [
        f"# Product: {profile.name}",
        f"Description: {profile.raw_description}",
    ]

    if profile.price is not None:
        period_label = {
            "annual": "/yr",
            "one-time": " one-time",
            "monthly": "/mo",
        }.get(profile.billing_period, "/mo")
        parts.append(f"Price: ${profile.price}{period_label}")
    if profile.price_model:
        parts.append(f"Price model: {profile.price_model}")
    if profile.value_proposition:
        parts.append(f"Value proposition: {profile.value_proposition}")
    if profile.target_audience:
        parts.append(f"Target audience: {profile.target_audience}")
    if profile.category:
        parts.append(f"Category: {profile.category}")

    # Competitor info
    all_competitors = profile.competitors[:]
    if market.verified_competitors:
        parts.append("\n## Verified Competitors (from research)")
        for c in market.verified_competitors:
            price_str = f"${c.confirmed_price}" if c.confirmed_price else "unknown price"
            parts.append(f"- {c.name}: {price_str}")
            if c.has_free_tier:
                parts.append(f"  (has free tier)")

    if market.discovered_competitors:
        parts.append("\n## Discovered Competitors (not mentioned by user)")
        for c in market.discovered_competitors:
            price_str = f"${c.confirmed_price}" if c.confirmed_price else "unknown price"
            parts.append(f"- {c.name}: {price_str}")

    # Market data
    if market.category_penetration is not None:
        parts.append(f"\nCategory penetration: {market.category_penetration*100:.1f}%")
    if market.category_growth is not None:
        parts.append(f"Category growth: {market.category_growth}")
    if market.category_maturity:
        parts.append(f"Category maturity: {market.category_maturity}")

    if market.low_confidence_flags:
        parts.append(f"\nLow confidence flags: {', '.join(market.low_confidence_flags)}")

    if market.research_skipped:
        parts.append("\n⚠️ Research was skipped. Use behavioral science defaults.")

    return "\n".join(parts)


def _parse_calibration_response(
    data: dict,
    profile: ProductProfile,
) -> SimulationConfig:
    """Parse LLM calibration response into SimulationConfig."""

    def _parse_cparam(d: dict | float | None, default: float = 0.5) -> CalibratedParam:
        if d is None:
            return CalibratedParam(value=default)
        if isinstance(d, (int, float)):
            return CalibratedParam(value=float(d))
        return CalibratedParam(
            value=float(d.get("value", default)),
            basis=d.get("basis", ""),
            confidence=float(d.get("confidence", 0.5)),
        )

    # Reference price
    ref_data = data.get("reference_price", {})
    if isinstance(ref_data, (int, float)):
        ref_price = ReferencePriceParam(value=float(ref_data))
    else:
        components = [
            ReferencePriceComponent(
                source=c.get("source", ""),
                price=float(c.get("price", 0)),
                weight=float(c.get("weight", 0)),
            )
            for c in ref_data.get("components", [])
        ]
        ref_price = ReferencePriceParam(
            value=float(ref_data.get("value", 0)),
            components=components,
            confidence=float(ref_data.get("confidence", 0.5)),
            basis=ref_data.get("basis", ""),
        )

    # Awareness
    aware_data = data.get("awareness_by_archetype", {})
    awareness = AwarenessByArchetype(
        innovator=float(aware_data.get("innovator", 0.90)),
        early_adopter=float(aware_data.get("early_adopter", 0.65)),
        early_majority=float(aware_data.get("early_majority", 0.30)),
        late_majority=float(aware_data.get("late_majority", 0.10)),
        laggard=float(aware_data.get("laggard", 0.02)),
    )

    # Population config
    pop_data = data.get("population_config", {})
    pop_config = PopulationConfig(
        income_mean_log=float(pop_data.get("income_mean_log", 11.0)),
        income_sigma=float(pop_data.get("income_sigma", 0.7)),
        market_segment=pop_data.get("market_segment", "general"),
    )

    # Assumptions
    assumptions = []
    for i, a in enumerate(data.get("assumptions", []), 1):
        assumptions.append(Assumption(
            id=f"A{i}",
            parameter=a.get("parameter", ""),
            value=float(a.get("value", 0)),
            basis=a.get("basis", ""),
            confidence=float(a.get("confidence", 0.5)),
            sensitivity=a.get("sensitivity", "medium"),
        ))

    sim_params = SimulationParams(
        price=float(data.get("price", profile.price or 0)),
        reference_price=ref_price,
        category_penetration=_parse_cparam(data.get("category_penetration"), 0.15),
        category_growth=_parse_cparam(data.get("category_growth"), 0.5),
        benefit_certainty=_parse_cparam(data.get("benefit_certainty"), 0.5),
        perceived_benefit=_parse_cparam(data.get("perceived_benefit"), 0.6),
        time_to_value=_parse_cparam(data.get("time_to_value"), 0.3),
        requires_behavior_change=_parse_cparam(data.get("requires_behavior_change"), 0.3),
        switching_cost=_parse_cparam(data.get("switching_cost"), 0.4),
        social_visibility=_parse_cparam(data.get("social_visibility"), 0.5),
        identity_signal=_parse_cparam(data.get("identity_signal"), 0.3),
        present_bias_beta=_parse_cparam(data.get("present_bias_beta"), 0.75),
        fomo_intensity=_parse_cparam(data.get("fomo_intensity"), 0.4),
        product_adoption_rate=_parse_cparam(data.get("product_adoption_rate"), 0.05),
        awareness=awareness,
    )

    return SimulationConfig(
        simulation_params=sim_params,
        population_config=pop_config,
        assumptions=assumptions,
    )
