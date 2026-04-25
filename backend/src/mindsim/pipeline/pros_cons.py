"""Wave 8 — pros/cons triangulation enforcer.

Each ProsConsItem must be supported by BOTH a simulation signal AND a
VoC signal to land in the report as a `pro` or `con`. Single-source
claims drop to `polarity="nuance"`. The narrative author (A7) consumes
triangulated items as load-bearing claims and nuance items as
"directional, not load-bearing" disclaimers.

This file implements the matching logic in pure Python — no LLM —
because a hard rule is more useful than a re-promptable judgement
call. Wave 8.5 layers `evidence_strength` counts on top of what we
emit here.

Matching strategy:
  1. Extract simulation signals — top 2 drivers, top 2 blockers from
     `SimulationResult.force_decomposition`.
  2. Extract VoC signals — pain_points / delight_points (each carries
     its quote_ids when feature_sentiment refers to them).
  3. Keyword-overlap a force name with a VoC theme — if a force keyword
     appears in a pain_point's text and the force value is negative,
     that's a triangulated `con`. Same for delights → drivers → `pro`.
  4. Sim signals with no VoC support → `nuance` (mechanism = "simulation only").
  5. VoC signals with no sim support → `nuance` (mechanism = "VoC only").

The keyword dictionary is intentionally small and conservative — it's
better to over-flag nuance than to claim triangulation that doesn't
exist. Wave 8.5 plans an `evidence_strength` count to make the
quality of each match visible.
"""
from __future__ import annotations

import logging
from typing import Iterable

from mindsim.models.results import EvidenceStrength, ProsConsItem, SimulationResult

logger = logging.getLogger(__name__)


# Force → keywords commonly used in customer voice for that mechanism.
# Keep terms short; the matcher does substring containment, lowercased.
FORCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "prospect_value": (
        "value", "benefit", "useful", "worth", "powerful", "helpful", "feature",
        "missing", "lacks",
    ),
    "anchoring": (
        "price", "expensive", "cheap", "cost", "afford", "pricing", "overpriced",
        "fee",
    ),
    "status_quo": (
        "switch", "migration", "migrate", "habit", "incumbent", "stuck", "lock",
        "leave", "incumbent", "status quo",
    ),
    "social_proof": (
        "everyone", "popular", "trend", "mainstream", "community", "team",
        "everyone uses",
    ),
    "fomo": (
        "miss", "behind", "fall behind", "fomo", "left out", "missing out",
    ),
    "hyperbolic_discounting": (
        "learning curve", "delay", "wait", "ramp", "onboard", "slow", "later",
        "set up", "setup",
    ),
    "identity_signaling": (
        "image", "look", "status", "premium", "brand", "professional", "prestige",
        "credibility",
    ),
}


# How many top forces to mine from each side of the dominance ranking.
TOP_DRIVERS = 2
TOP_BLOCKERS = 2


def build_pros_cons(
    sim_result: SimulationResult,
    voc,
) -> list[ProsConsItem]:
    """Triangulate simulation force dominance against VoC themes.

    Args:
        sim_result: post-simulate SimulationResult with force_decomposition.
        voc: VoCReport (or None / object-without-fields when VoC was
             skipped). When the VoC corpus is empty, every sim signal
             collapses to a `nuance` (no triangulation possible).

    Returns:
        A list of ProsConsItem in the order:
          1. Triangulated pros (force driver + matching VoC delight)
          2. Triangulated cons (force blocker + matching VoC pain)
          3. Nuances (sim-only or VoC-only signals)
        The narrative author consumes them in this order.
    """
    forces = sim_result.force_decomposition.as_dict()
    pain_points = list(getattr(voc, "pain_points", []) or [])
    delight_points = list(getattr(voc, "delight_points", []) or [])
    feature_sentiment = list(getattr(voc, "feature_sentiment", []) or [])

    drivers = sorted(forces.items(), key=lambda kv: kv[1], reverse=True)[:TOP_DRIVERS]
    blockers = sorted(forces.items(), key=lambda kv: kv[1])[:TOP_BLOCKERS]

    triangulated_pros: list[ProsConsItem] = []
    triangulated_cons: list[ProsConsItem] = []
    nuances: list[ProsConsItem] = []

    # Track which VoC themes we've consumed for triangulation. Anything
    # untouched at the end becomes a VoC-only nuance.
    matched_pain_idx: set[int] = set()
    matched_delight_idx: set[int] = set()

    # ── Drivers → triangulated pros (when matching VoC delight) ──
    for force_name, force_value in drivers:
        if force_value <= 0:
            # Not actually a driver — top force is still net negative.
            # Emit as a sim-only nuance.
            nuances.append(_sim_only_nuance(force_name, force_value, polarity_hint="con"))
            continue
        matches = _match_voc(force_name, delight_points)
        if matches:
            quote_ids = _collect_quote_ids(force_name, feature_sentiment)
            triangulated_pros.append(ProsConsItem(
                statement=_pro_statement(force_name, matches[0]),
                polarity="pro",
                segments=[],
                mechanism=f"{force_name}={force_value:+.3f} (sim) corroborated by VoC delight",
                simulation_evidence={"force": force_name, "value": float(force_value)},
                voc_evidence=quote_ids,
            ))
            for idx in _theme_indices(matches, delight_points):
                matched_delight_idx.add(idx)
        else:
            nuances.append(_sim_only_nuance(force_name, force_value, polarity_hint="pro"))

    # ── Blockers → triangulated cons (when matching VoC pain) ──
    for force_name, force_value in blockers:
        if force_value >= 0:
            nuances.append(_sim_only_nuance(force_name, force_value, polarity_hint="pro"))
            continue
        matches = _match_voc(force_name, pain_points)
        if matches:
            quote_ids = _collect_quote_ids(force_name, feature_sentiment)
            triangulated_cons.append(ProsConsItem(
                statement=_con_statement(force_name, matches[0]),
                polarity="con",
                segments=[],
                mechanism=f"{force_name}={force_value:+.3f} (sim) corroborated by VoC pain",
                simulation_evidence={"force": force_name, "value": float(force_value)},
                voc_evidence=quote_ids,
            ))
            for idx in _theme_indices(matches, pain_points):
                matched_pain_idx.add(idx)
        else:
            nuances.append(_sim_only_nuance(force_name, force_value, polarity_hint="con"))

    # ── VoC themes with no sim match → VoC-only nuances ──
    for i, pp in enumerate(pain_points):
        if i in matched_pain_idx:
            continue
        nuances.append(_voc_only_nuance(pp, polarity_hint="con"))
    for i, dp in enumerate(delight_points):
        if i in matched_delight_idx:
            continue
        nuances.append(_voc_only_nuance(dp, polarity_hint="pro"))

    items = triangulated_pros + triangulated_cons + nuances
    for item in items:
        item.evidence_strength = _classify_strength(item)
    return items


def _classify_strength(item: ProsConsItem) -> EvidenceStrength:
    """Mechanical Wave 8.5 classification: triangulated / nuance / weak.

    n_sim = 1 if simulation_evidence carries any payload, else 0.
    n_voc = number of quote_ids attached.
    cls   = triangulated when both sides ≥ 1, nuance when exactly one,
            weak when zero (validator should block these from reaching
            the narrative author).
    """
    n_sim = 1 if (item.simulation_evidence or {}) else 0
    n_voc = len(item.voc_evidence or [])
    if n_sim >= 1 and n_voc >= 1:
        cls = "triangulated"
    elif n_sim >= 1 or n_voc >= 1:
        cls = "nuance"
    else:
        cls = "weak"
    return EvidenceStrength(n_sim=n_sim, n_voc=n_voc, cls=cls)


# ─────────────────────── matching helpers ───────────────────────


def _match_voc(force_name: str, themes: Iterable[str]) -> list[str]:
    """Return themes whose lowercased text contains a force keyword."""
    keywords = FORCE_KEYWORDS.get(force_name, ())
    if not keywords:
        return []
    out: list[str] = []
    for theme in themes:
        text = (theme or "").lower()
        if not text:
            continue
        if any(k in text for k in keywords):
            out.append(theme)
    return out


def _theme_indices(matches: list[str], all_themes: list[str]) -> list[int]:
    """Indices in `all_themes` corresponding to entries in `matches`."""
    return [i for i, t in enumerate(all_themes) if t in matches]


def _collect_quote_ids(
    force_name: str,
    feature_sentiments: list,
) -> list[str]:
    """Pull quote_ids from feature_sentiment rows whose feature name overlaps
    with the force's keyword set.

    Best-effort — many VoCReports won't have feature-keyed sentiment,
    in which case we return an empty list and the resulting evidence
    record stays sparse. Wave 8.5 makes evidence depth visible via
    `evidence_strength` counts.
    """
    keywords = FORCE_KEYWORDS.get(force_name, ())
    if not keywords:
        return []
    out: list[str] = []
    for fs in feature_sentiments:
        feature_name = (getattr(fs, "feature", "") or "").lower()
        if any(k in feature_name for k in keywords):
            out.extend(getattr(fs, "quote_ids", []) or [])
    # Dedupe while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for q in out:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


def _pro_statement(force_name: str, voc_theme: str) -> str:
    return f"Sim driver `{force_name}` aligns with VoC delight: \"{voc_theme}\""


def _con_statement(force_name: str, voc_theme: str) -> str:
    return f"Sim blocker `{force_name}` aligns with VoC pain: \"{voc_theme}\""


def _sim_only_nuance(force_name: str, value: float, polarity_hint: str) -> ProsConsItem:
    return ProsConsItem(
        statement=(
            f"Sim {polarity_hint}: `{force_name}` averages {value:+.3f} but no "
            f"matching VoC signal — directional, not load-bearing."
        ),
        polarity="nuance",
        mechanism="simulation only",
        simulation_evidence={"force": force_name, "value": float(value)},
        voc_evidence=[],
    )


def _voc_only_nuance(theme: str, polarity_hint: str) -> ProsConsItem:
    return ProsConsItem(
        statement=(
            f"VoC {polarity_hint}: \"{theme}\" — no matching simulation force, "
            f"directional only."
        ),
        polarity="nuance",
        mechanism="VoC only",
        simulation_evidence={},
        voc_evidence=[],
    )
