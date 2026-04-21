"""Stage A2 — Research Synthesizer.

Consumes Tavily snippets + scraped pricing pages + VoCReport → produces
a synthesized `MarketContext` with competitor feature_scores,
market_share, and stable `source_ids` citations.

One LLM call (`SYNTHESIZER_PROMPT`). Defensive parsing: fabricated
source_ids (not in the supplied evidence set) are dropped; competitors
with no evidence are discarded.
"""
from __future__ import annotations

import logging

from mindsim.llm.client import LLMClient
from mindsim.llm.prompts import SYNTHESIZER_PROMPT
from mindsim.models.market import MarketContext, VerifiedCompetitor
from mindsim.models.product import ProductProfile
from mindsim.models.voc import VoCReport
from mindsim.scrape.base import ScrapedDocument, make_source_id

logger = logging.getLogger(__name__)


_FEATURE_CATEGORIES = (
    "core_value",
    "social_signal",
    "ongoing_cost",
    "switching_friction_reducer",
)


def synthesize_market(
    context: MarketContext,
    tavily_results: list[dict],
    scraped_pricing_docs: list[ScrapedDocument],
    voc: VoCReport,
    profile: ProductProfile,
    llm_client: LLMClient | None = None,
) -> MarketContext:
    """Extend `context` with synthesized competitor matrix.

    Mutates and returns `context` so downstream callers can keep the same
    reference. Never raises — LLM / parse failure means the context is
    returned unchanged except for a low_confidence_flags entry.
    """
    if not tavily_results and not scraped_pricing_docs and not voc.quotes:
        logger.info("A2 synthesizer: no evidence available, skipping")
        context.low_confidence_flags.append(
            "A2 synthesizer skipped — no research evidence available."
        )
        return context

    # Assemble the valid source_id set the LLM is allowed to cite.
    source_catalog: dict[str, dict] = {}
    for t in tavily_results or []:
        url = t.get("url") or ""
        snippet = (t.get("content") or t.get("title") or "")[:240]
        if not url:
            continue
        sid = make_source_id(url, snippet)
        source_catalog[sid] = {"url": url, "snippet": snippet, "kind": "tavily"}
    for d in scraped_pricing_docs or []:
        sid = make_source_id(d.url, d.body[:240])
        source_catalog[sid] = {
            "url": d.url,
            "snippet": d.body[:240],
            "kind": "pricing_page",
            "metadata": d.metadata,
        }

    user_message = _build_user_message(source_catalog, voc, profile)

    client = llm_client or LLMClient()
    try:
        raw = client.complete_json(SYNTHESIZER_PROMPT, user_message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("A2 synthesizer LLM call failed: %s", exc)
        context.low_confidence_flags.append(
            f"A2 synthesizer failed: {type(exc).__name__}"
        )
        return context

    competitors = _parse_competitors(raw, source_catalog)
    # Merge — new rows stamp into verified_competitors (preserve existing names).
    existing_names = {c.name for c in context.verified_competitors}
    for comp in competitors:
        if comp.name in existing_names:
            continue
        context.verified_competitors.append(comp)

    # Category penetration override (if A2 is more confident than Phase 1).
    pen = raw.get("category_penetration")
    pen_conf = raw.get("category_penetration_confidence")
    if pen is not None:
        try:
            pen_f = float(pen)
            pen_conf_f = float(pen_conf or 0.0)
            if 0.0 <= pen_f <= 1.0 and pen_conf_f > context.category_penetration_confidence:
                context.category_penetration = pen_f
                context.category_penetration_confidence = pen_conf_f
        except (TypeError, ValueError):
            pass

    return context


# ───────────────────────── Internals ─────────────────────────


def _build_user_message(
    source_catalog: dict[str, dict],
    voc: VoCReport,
    profile: ProductProfile,
) -> str:
    lines = [
        f"Product under analysis: {profile.name}",
        f"Category: {profile.category or 'unknown'}",
        f"Value proposition: {profile.value_proposition or 'n/a'}",
        "",
        "Valid source_ids (cite only these):",
    ]
    for sid, info in source_catalog.items():
        snippet = info.get("snippet", "")[:200].replace("\n", " ")
        lines.append(f"  [{sid}] ({info['kind']}) {info['url']}\n    {snippet}")

    lines.append("")
    lines.append("VoC pain points:")
    for p in voc.pain_points[:10]:
        lines.append(f"  - {p}")
    lines.append("VoC delight points:")
    for p in voc.delight_points[:10]:
        lines.append(f"  - {p}")
    if voc.bias_note:
        lines.append(f"(VoC bias note: {voc.bias_note})")

    return "\n".join(lines)


def _parse_competitors(
    raw: dict,
    source_catalog: dict[str, dict],
) -> list[VerifiedCompetitor]:
    competitors_raw = raw.get("competitors", []) or []
    valid_sids = set(source_catalog.keys())
    out: list[VerifiedCompetitor] = []
    for c in competitors_raw:
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").strip()
        if not name:
            continue

        # Filter source_ids to only real ones.
        sids = [s for s in (c.get("source_ids") or []) if s in valid_sids]
        if not sids:
            # No evidence — drop to preserve provenance invariant.
            logger.debug("dropping competitor %r: no valid source_ids", name)
            continue

        feature_scores_raw = c.get("feature_scores") or {}
        feature_scores: dict[str, float] = {}
        for cat in _FEATURE_CATEGORIES:
            v = feature_scores_raw.get(cat)
            if v is None:
                continue
            try:
                feature_scores[cat] = max(0.0, min(float(v), 1.0))
            except (TypeError, ValueError):
                continue

        market_share = c.get("market_share")
        try:
            market_share_f = (
                float(market_share) if market_share is not None else None
            )
            if market_share_f is not None and not (0.0 <= market_share_f <= 1.0):
                market_share_f = None
        except (TypeError, ValueError):
            market_share_f = None

        confirmed_price = c.get("confirmed_price")
        try:
            price_f = float(confirmed_price) if confirmed_price is not None else None
            if price_f is not None and price_f < 0:
                price_f = None
        except (TypeError, ValueError):
            price_f = None

        out.append(
            VerifiedCompetitor(
                name=name[:80],
                confirmed_price=price_f,
                price_model=c.get("price_model"),
                has_free_tier=bool(c.get("has_free_tier")),
                market_position=c.get("market_position"),
                source_url=c.get("source_url") or source_catalog[sids[0]]["url"],
                feature_scores=feature_scores,
                market_share=market_share_f,
                source_ids=sids,
            )
        )
    return out
