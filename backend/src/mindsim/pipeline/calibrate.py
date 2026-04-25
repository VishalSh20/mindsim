"""Stage 3: CALIBRATE — assign ALL numerical simulation parameters.

Takes ProductProfile + MarketContext → calls LLM → parses into SimulationConfig.

v2-middle Wave 2: the LLM now emits a `feature_matrix` (4-8 Feature objects)
instead of scalar `perceived_benefit` / `benefit_certainty` / `switching_cost`
/ `social_visibility` / `time_to_value`. Those five scalars are *derived*
from the feature matrix inside this stage so downstream code (forces.py,
sensitivity analysis, display) still reads them. Each derived scalar gets
per-archetype overrides automatically, since each archetype has its own
category weights in `config/feature_weights.yaml`.
"""

from __future__ import annotations

import logging
from typing import Iterable

from mindsim.engine.feature_weights import (
    FEATURE_CATEGORIES,
    load_feature_weights,
)
from mindsim.llm.client import LLMClient
from mindsim.llm.prompts import CALIBRATE_PROMPT
from mindsim.models.config import (
    ARCHETYPE_NAMES,
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
from mindsim.models.product import Feature, ProductProfile

logger = logging.getLogger(__name__)


VALID_ARCHETYPES = set(ARCHETYPE_NAMES)

# Fallback feature matrix used when the LLM fails to emit one (e.g., parse
# failure on --skip-research). Keeps the simulation deterministic rather
# than silently producing degenerate adoption numbers.
DEFAULT_FEATURE_MATRIX: list[Feature] = [
    Feature(
        name="core_functionality",
        score=0.60,
        polarity="positive",
        category="core_value",
        certainty=0.50,
        visibility=0.30,
        time_to_value_months=1.0,
        basis="default (LLM did not emit a feature_matrix)",
    ),
    Feature(
        name="ongoing_friction",
        score=0.40,
        polarity="negative",
        category="ongoing_cost",
        certainty=0.60,
        visibility=0.10,
        time_to_value_months=0.0,
        basis="default",
    ),
    Feature(
        name="integrations",
        score=0.50,
        polarity="positive",
        category="switching_friction_reducer",
        certainty=0.60,
        visibility=0.20,
        time_to_value_months=0.0,
        basis="default",
    ),
    Feature(
        name="visible_usage",
        score=0.30,
        polarity="positive",
        category="social_signal",
        certainty=0.50,
        visibility=0.50,
        time_to_value_months=0.5,
        basis="default",
    ),
]


def calibrate(
    profile: ProductProfile,
    market: MarketContext,
    llm: LLMClient,
) -> SimulationConfig:
    """Assign all simulation parameters via LLM calibration."""
    logger.info("Stage 3: Calibrating simulation parameters...")

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
        f"features={len(config.simulation_params.feature_matrix)}, "
        f"β={config.simulation_params.present_bias_beta.value:.2f}"
    )
    return config


def _build_calibration_context(
    profile: ProductProfile,
    market: MarketContext,
) -> str:
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


# ───────────────────────── helpers ─────────────────────────


def _parse_by_archetype(d: dict) -> dict[str, float] | None:
    raw = d.get("by_archetype")
    if raw is None or not isinstance(raw, dict):
        return None
    filtered = {
        k: float(v)
        for k, v in raw.items()
        if k in VALID_ARCHETYPES and isinstance(v, (int, float))
    }
    return filtered if filtered else None


def _parse_cparam(d: dict | float | None, default: float = 0.5) -> CalibratedParam:
    if d is None:
        return CalibratedParam(value=default)
    if isinstance(d, (int, float)):
        return CalibratedParam(value=float(d))
    refs = d.get("evidence_refs")
    refs_list = [str(r) for r in refs] if isinstance(refs, list) else []
    src = d.get("source_type")
    src_str = str(src) if src in {"voc", "research", "trends", "default", "llm_judgment"} else None
    return CalibratedParam(
        value=float(d.get("value", default)),
        basis=d.get("basis", ""),
        confidence=float(d.get("confidence", 0.5)),
        by_archetype=_parse_by_archetype(d),
        evidence_refs=refs_list,
        source_type=src_str,
    )


def _parse_feature_matrix(data: dict) -> list[Feature]:
    raw = data.get("feature_matrix", [])
    if not isinstance(raw, list):
        logger.warning("feature_matrix is not a list; using defaults")
        return list(DEFAULT_FEATURE_MATRIX)

    features: list[Feature] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        try:
            features.append(
                Feature(
                    name=str(item.get("name", f"feature_{i}")),
                    score=float(item.get("score", 0.5)),
                    polarity=item.get("polarity", "positive"),
                    category=item.get("category", "core_value"),
                    certainty=float(item.get("certainty", 0.5)),
                    visibility=float(item.get("visibility", 0.3)),
                    time_to_value_months=float(item.get("time_to_value_months", 0.0)),
                    basis=str(item.get("basis", "")),
                    evidence_refs=(
                        list(item["evidence_refs"])
                        if isinstance(item.get("evidence_refs"), list)
                        else []
                    ),
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("skipping malformed feature %d: %s", i, exc)

    if not features:
        logger.warning("no valid features parsed; using defaults")
        return list(DEFAULT_FEATURE_MATRIX)
    return features


def _parse_competitor_feature_scores(data: dict) -> dict[str, dict[str, float]]:
    raw = data.get("competitor_feature_scores", {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for name, scores in raw.items():
        if not isinstance(scores, dict):
            continue
        cleaned = {
            k: float(v)
            for k, v in scores.items()
            if isinstance(v, (int, float))
        }
        if cleaned:
            out[str(name)] = cleaned
    return out


def _derive_aggregate_scalars(features: list[Feature]) -> dict[str, CalibratedParam]:
    """Derive the 5 aggregate scalars from the feature matrix.

    `perceived_benefit`, `benefit_certainty`, `switching_cost`,
    `social_visibility`, `time_to_value` are all computed per-archetype
    using that archetype's category weights. The early_majority value
    becomes the base; the other four archetypes land as `by_archetype`
    overrides — so the existing population-stamp logic picks them up
    without modification.

    `time_to_value` is kept on the legacy [0, 1] scale (months/12 clipped)
    so Wave 1 force math still works.
    """
    try:
        fw = load_feature_weights()
    except Exception as exc:  # noqa: BLE001
        logger.warning("feature_weights YAML unreadable (%s); using uniform 0.25 weights", exc)
        uniform = {c: 0.25 for c in FEATURE_CATEGORIES}
        fw_weights = {a: dict(uniform) for a in ARCHETYPE_NAMES}
    else:
        fw_weights = {a: fw.weights_for(a) for a in ARCHETYPE_NAMES}

    # For each archetype, compute: perceived_benefit, benefit_certainty,
    # switching_cost, social_visibility, time_to_value (normalised).
    per_archetype: dict[str, dict[str, float]] = {a: {} for a in ARCHETYPE_NAMES}

    pos = [f for f in features if f.polarity == "positive"]
    neg = [f for f in features if f.polarity == "negative"]
    sfr = [f for f in features if f.category == "switching_friction_reducer"]

    for a in ARCHETYPE_NAMES:
        w = fw_weights[a]

        if pos:
            tw = sum(w[f.category] for f in pos) or 1.0
            pb = sum(w[f.category] * f.score for f in pos) / tw
        else:
            pb = 0.25

        if features:
            tw_all = sum(w[f.category] for f in features) or 1.0
            bc = sum(w[f.category] * f.certainty for f in features) / tw_all
        else:
            bc = 0.5

        # Switching cost inversely related to switching_friction_reducer strength
        if sfr:
            tw_sfr = sum(w[f.category] for f in sfr) or 1.0
            sfr_weighted = sum(w[f.category] * f.score for f in sfr) / tw_sfr
            sc = max(0.2, 1.0 - 0.7 * sfr_weighted)
        else:
            sc = 0.7

        if pos:
            tw = sum(w[f.category] for f in pos) or 1.0
            sv = sum(w[f.category] * f.visibility for f in pos) / tw
        else:
            sv = 0.3

        if pos:
            tw = sum(w[f.category] for f in pos) or 1.0
            ttv_months = sum(w[f.category] * f.time_to_value_months for f in pos) / tw
        else:
            ttv_months = 0.0
        ttv_normalised = min(1.0, ttv_months / 12.0)

        per_archetype[a] = {
            "perceived_benefit": round(max(0.0, min(1.0, pb)), 4),
            "benefit_certainty": round(max(0.0, min(1.0, bc)), 4),
            "switching_cost": round(max(0.0, min(1.0, sc)), 4),
            "social_visibility": round(max(0.0, min(1.0, sv)), 4),
            "time_to_value": round(max(0.0, min(1.0, ttv_normalised)), 4),
        }

    # Use early_majority as the base value; other 4 archetypes → by_archetype.
    base_key = "early_majority"
    out: dict[str, CalibratedParam] = {}
    for field in ("perceived_benefit", "benefit_certainty", "switching_cost",
                  "social_visibility", "time_to_value"):
        base_value = per_archetype[base_key][field]
        overrides = {
            a: per_archetype[a][field]
            for a in ARCHETYPE_NAMES
            if a != base_key
        }
        out[field] = CalibratedParam(
            value=base_value,
            basis=f"derived from feature_matrix ({base_key} as base, per-archetype overrides applied)",
            confidence=0.7,
            by_archetype=overrides,
        )
    return out


# ───────────────────────── main parser ─────────────────────────


def _parse_calibration_response(
    data: dict,
    profile: ProductProfile,
) -> SimulationConfig:
    """Parse LLM calibration response into SimulationConfig."""

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
            for c in (ref_data.get("components", []) if isinstance(ref_data, dict) else [])
        ]
        ref_price = ReferencePriceParam(
            value=float(ref_data.get("value", 0)) if isinstance(ref_data, dict) else 0.0,
            components=components,
            confidence=float(ref_data.get("confidence", 0.5)) if isinstance(ref_data, dict) else 0.5,
            basis=ref_data.get("basis", "") if isinstance(ref_data, dict) else "",
            by_archetype=_parse_by_archetype(ref_data) if isinstance(ref_data, dict) else None,
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
        a_refs = a.get("evidence_refs")
        a_refs_list = [str(r) for r in a_refs] if isinstance(a_refs, list) else []
        a_src = a.get("source_type")
        a_src_str = (
            str(a_src)
            if a_src in {"voc", "research", "trends", "default", "llm_judgment"}
            else None
        )
        assumptions.append(
            Assumption(
                id=f"A{i}",
                parameter=a.get("parameter", ""),
                value=float(a.get("value", 0)),
                basis=a.get("basis", ""),
                confidence=float(a.get("confidence", 0.5)),
                sensitivity=a.get("sensitivity", "medium"),
                evidence_refs=a_refs_list,
                source_type=a_src_str,
            )
        )

    # v2-middle Wave 2: feature matrix + derived aggregates
    feature_matrix = _parse_feature_matrix(data)
    competitor_feature_scores = _parse_competitor_feature_scores(data)
    derived = _derive_aggregate_scalars(feature_matrix)

    sim_params = SimulationParams(
        price=float(data.get("price", profile.price or 0)),
        reference_price=ref_price,
        category_penetration=_parse_cparam(data.get("category_penetration"), 0.15),
        category_growth=_parse_cparam(data.get("category_growth"), 0.5),
        # Derived-from-feature-matrix aggregates (not LLM-emitted any more):
        benefit_certainty=derived["benefit_certainty"],
        perceived_benefit=derived["perceived_benefit"],
        time_to_value=derived["time_to_value"],
        switching_cost=derived["switching_cost"],
        social_visibility=derived["social_visibility"],
        # Still LLM-emitted (product-level, not feature-level):
        requires_behavior_change=_parse_cparam(data.get("requires_behavior_change"), 0.3),
        identity_signal=_parse_cparam(data.get("identity_signal"), 0.3),
        present_bias_beta=_parse_cparam(data.get("present_bias_beta"), 0.75),
        fomo_intensity=_parse_cparam(data.get("fomo_intensity"), 0.4),
        product_adoption_rate=_parse_cparam(data.get("product_adoption_rate"), 0.05),
        awareness=awareness,
        feature_matrix=feature_matrix,
        competitor_feature_scores=competitor_feature_scores,
    )

    return SimulationConfig(
        simulation_params=sim_params,
        population_config=pop_config,
        assumptions=assumptions,
    )
