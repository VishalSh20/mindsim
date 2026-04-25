"""Stage 5: ANALYZE — sensitivity + confidence bands + behavioral audit.

Runs sensitivity analysis, computes confidence bands, ranks interventions,
calls LLM for behavioral audit prose, and assembles the final AnalysisReport.
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np

from mindsim.engine.interventions import rank_interventions
from mindsim.engine.sensitivity import run_sensitivity_analysis
from mindsim.llm.client import LLMClient
from mindsim.llm.prompts import ANALYZE_PROMPT
from mindsim.models.config import SimulationConfig
from mindsim.models.results import (
    AnalysisReport,
    ConfidenceBand,
    SimulationResult,
)
from mindsim.models.evidence import build_evidence_store_from_market
from mindsim.pipeline.kpi import build_kpi_dashboard
from mindsim.pipeline.pros_cons import build_pros_cons
from mindsim.pipeline.provenance import (
    build_parameter_reasoning_trail,
    collect_referenced_evidence_ids,
)
from mindsim.pipeline.segments import build_segment_report
from mindsim.pipeline.validator import populate_published_bounds

logger = logging.getLogger(__name__)


def analyze(
    sim_result: SimulationResult,
    config: SimulationConfig,
    llm: LLMClient,
    rng: np.random.Generator | None = None,
    market: "MarketContext | None" = None,
) -> AnalysisReport:
    """Run full analysis: sensitivity + confidence + interventions + Wave 8 KPI / segments / pros-cons + audit.

    Args:
        sim_result: Results from the simulate stage.
        config: SimulationConfig with parameter provenance.
        llm: LLM client for behavioral audit.
        rng: Random number generator.
        market: Optional MarketContext — when provided, its `voc_report`
            is consumed by the Wave 8 pros/cons triangulator. Without it
            every pros/cons claim degrades to `nuance`.

    Returns:
        Complete AnalysisReport with Wave 8 KPI dashboard, segment
        narratives, and triangulated pros/cons attached.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    agents = sim_result._agents
    if agents is None:
        raise ValueError("SimulationResult missing agent data for analysis")

    # 5a. SENSITIVITY ANALYSIS
    logger.info("Running sensitivity analysis...")
    sensitivity = run_sensitivity_analysis(
        agents=agents,
        config=config,
        base_adoption=sim_result.total_adoption,
        rng=rng,
    )

    # 5b. CONFIDENCE BAND (RSS of swings)
    confidence_band = _compute_confidence_band(
        central=sim_result.total_adoption,
        sensitivity_results=sensitivity,
    )

    # 5c. INTERVENTION RANKING
    logger.info("Ranking interventions...")
    interventions = rank_interventions(
        agents=agents,
        config=config,
        base_adoption=sim_result.total_adoption,
        rng=rng,
    )

    # 5d. WAVE 8 — KPI DASHBOARD + SEGMENT NARRATIVES + PROS/CONS
    logger.info("Building KPI dashboard / segments / triangulated pros-cons...")
    kpi = build_kpi_dashboard(sim_result, sensitivity_results=sensitivity)
    segments = build_segment_report(sim_result)
    voc = getattr(market, "voc_report", None) if market is not None else None
    pros_cons = build_pros_cons(sim_result, voc)

    # 5d.1 WAVE 8.5 — provenance + EvidenceStore.
    # `populate_published_bounds` mutates assumptions; safe to call before
    # the trail builder since it doesn't change values, only attaches bounds.
    populate_published_bounds(config)
    full_store = build_evidence_store_from_market(market, voc)
    referenced = collect_referenced_evidence_ids(config, pros_cons)
    # Compress: only retain IDs that some artefact actually cites. Anything
    # else is dropped so the dump doesn't bloat with the full corpus.
    evidence_store = full_store.compressed(referenced) if referenced else full_store
    sim_result.evidence_store = evidence_store
    reasoning_trail = build_parameter_reasoning_trail(config, evidence_store)

    # 5e. BEHAVIORAL AUDIT (LLM)
    logger.info("Generating behavioral audit...")
    audit_text = _generate_audit(
        sim_result=sim_result,
        config=config,
        sensitivity=sensitivity,
        confidence_band=confidence_band,
        interventions=interventions,
        llm=llm,
        kpi=kpi,
        segments=segments,
        pros_cons=pros_cons,
        voc=voc,
        reasoning_trail=reasoning_trail,
    )

    # Build assumptions list
    assumptions = [
        {
            "id": a.id,
            "parameter": a.parameter,
            "value": a.value,
            "basis": a.basis,
            "confidence": a.confidence,
            "sensitivity": a.sensitivity,
        }
        for a in config.assumptions
    ]

    report = AnalysisReport(
        text=audit_text,
        sensitivity=sensitivity,
        confidence_band=confidence_band,
        interventions=interventions,
        assumptions=assumptions,
        kpi=kpi,
        segments=segments,
        pros_cons=pros_cons,
    )

    logger.info("Analysis complete.")
    return report


def _compute_confidence_band(
    central: float,
    sensitivity_results: list,
) -> ConfidenceBand:
    """Compute confidence band from RSS of sensitivity swings."""
    if not sensitivity_results:
        return ConfidenceBand(
            central=central,
            low=max(0, central - 0.05),
            high=min(1, central + 0.05),
            drivers=[],
        )

    # RSS of all swings (convert from pp to fraction)
    swing_fractions = [s.swing / 100.0 for s in sensitivity_results]
    total_uncertainty = math.sqrt(sum(s ** 2 for s in swing_fractions))

    half_band = total_uncertainty / 2.0
    low = max(0.0, central - half_band)
    high = min(1.0, central + half_band)

    # Drivers: top contributors to uncertainty
    drivers = [s.parameter for s in sensitivity_results[:3]]

    return ConfidenceBand(
        central=central,
        low=low,
        high=high,
        drivers=drivers,
    )


def _generate_audit(
    sim_result: SimulationResult,
    config: SimulationConfig,
    sensitivity: list,
    confidence_band: ConfidenceBand,
    interventions: list,
    llm: LLMClient,
    kpi=None,
    segments=None,
    pros_cons: list | None = None,
    voc=None,
    reasoning_trail=None,
) -> str:
    """Generate behavioral audit prose via LLM."""
    # Build context for the LLM
    context_parts = [
        f"# Simulation Results",
        f"Total adoption: {sim_result.total_adoption*100:.1f}%",
        f"Aware adoption: {sim_result.aware_adoption*100:.1f}%",
        f"Agents: {sim_result.n_agents} total, {sim_result.n_aware} aware",
        f"Confidence band: {confidence_band.low*100:.0f}%–{confidence_band.high*100:.0f}%",
        "",
        "## Force Decomposition (convertible pool)",
    ]

    forces = sim_result.force_decomposition.as_dict()
    for name, val in forces.items():
        context_parts.append(f"  {name}: {val:+.3f}")

    context_parts.append("")
    context_parts.append("## Segments by Archetype")
    for seg in sim_result.by_archetype:
        context_parts.append(
            f"  {seg.name}: {seg.adoption_rate*100:.1f}% ({seg.count}/{seg.total})"
        )

    context_parts.append("")
    context_parts.append("## Segments by Income")
    for seg in sim_result.by_income:
        context_parts.append(
            f"  {seg.bracket}: {seg.adoption_rate*100:.1f}% ({seg.count}/{seg.total})"
        )

    context_parts.append("")
    context_parts.append("## Proximity")
    context_parts.append(
        f"  Locked: {sim_result.by_proximity.locked*100:.0f}% | "
        f"Convertible: {sim_result.by_proximity.convertible*100:.0f}% | "
        f"Unreachable: {sim_result.by_proximity.unreachable*100:.0f}%"
    )

    if sensitivity:
        context_parts.append("")
        context_parts.append("## Sensitivity Analysis")
        for s in sensitivity:
            context_parts.append(
                f"  {s.parameter}: {s.low_adoption*100:.0f}%–{s.high_adoption*100:.0f}% "
                f"(swing: {s.swing:.0f}pp, confidence: {s.confidence:.1f})"
            )

    if interventions:
        context_parts.append("")
        context_parts.append("## Interventions (ranked by cost-per-adoption-pp)")
        for intv in interventions:
            cost_str = (
                f"${intv.cost_usd_low/1000:.0f}-${intv.cost_usd_high/1000:.0f}k"
                if intv.cost_usd_high > 0
                else "free"
            )
            cpp_str = (
                f"${intv.cost_per_adoption_pp/1000:.1f}k/pp"
                if math.isfinite(intv.cost_per_adoption_pp)
                else "(no lift)"
            )
            tag = f" [{intv.additivity}]" if intv.additivity else ""
            context_parts.append(
                f"  {intv.name}: {intv.lift_pp:+.1f}pp over "
                f"{intv.timeline_rounds} rounds, {cost_str}, {cpp_str}{tag}"
            )

    if config.assumptions:
        context_parts.append("")
        context_parts.append("## Assumptions")
        for a in config.assumptions:
            context_parts.append(
                f"  {a.id} {a.parameter}: {a.value} (confidence: {a.confidence}, "
                f"basis: \"{a.basis}\")"
            )

    context_parts.append("")
    context_parts.append(f"## Product")
    context_parts.append(f"Price: ${config.simulation_params.price:.2f}/mo")
    context_parts.append(
        f"Reference price: ${config.simulation_params.reference_price.value:.2f}"
    )

    # ── Wave 8 additions ──
    if kpi is not None:
        context_parts.append("")
        context_parts.append("## KPI Dashboard")
        a = kpi.adoption
        context_parts.append(
            f"  Total adoption: {a.total*100:.1f}%, aware: {a.aware*100:.1f}%, "
            f"time-to-50%: {a.time_to_50pct or 'not reached'} round(s), "
            f"chasm: {a.chasm_round or 'no stall'}"
        )
        fd = kpi.force_dominance
        context_parts.append(
            f"  Top driver: {fd.top_driver}, top blocker: {fd.top_blocker}"
        )
        context_parts.append(f"  Convertible pool: {kpi.convertible_pool} agents")
        if kpi.cascade and kpi.cascade.first_cluster_crossed_critical_mass is not None:
            context_parts.append(
                f"  First cluster to critical mass: cluster "
                f"{kpi.cascade.first_cluster_crossed_critical_mass}"
            )
        if kpi.top_sensitivity_params:
            context_parts.append(
                f"  Top sensitivity params: {', '.join(kpi.top_sensitivity_params)}"
            )

    if segments is not None and segments.segments:
        context_parts.append("")
        context_parts.append("## Per-Segment Narratives")
        for s in segments.segments:
            context_parts.append(
                f"  {s.archetype}: {s.adoption_rate*100:.1f}% "
                f"({s.count}/{s.total}); driver={s.dominant_driver} "
                f"({s.dominant_driver_value:+.3f}), blocker={s.dominant_blocker} "
                f"({s.dominant_blocker_value:+.3f})"
            )

    if pros_cons:
        context_parts.append("")
        context_parts.append("## Triangulated Pros / Cons")
        triangulated = [p for p in pros_cons if p.polarity in ("pro", "con")]
        nuances = [p for p in pros_cons if p.polarity == "nuance"]
        for p in triangulated:
            context_parts.append(
                f"  [{p.polarity.upper()}] {p.statement} "
                f"({p.mechanism}; voc_refs={len(p.voc_evidence)})"
            )
        if nuances:
            context_parts.append("  --- Nuance (single-source, directional) ---")
            for p in nuances[:6]:  # cap to keep prompt tight
                context_parts.append(f"  [NUANCE] {p.statement}")

    if voc is not None and getattr(voc, "bias_note", None):
        context_parts.append("")
        context_parts.append(f"## VoC Bias Note\n  {voc.bias_note}")

    # ── Wave 8.5 — parameter reasoning trail with resolved evidence ──
    if reasoning_trail is not None and reasoning_trail.entries:
        context_parts.append("")
        context_parts.append("## Parameter Reasoning Trail (Wave 8.5)")
        for e in reasoning_trail.entries:
            cite = (
                f"[{', '.join(ev.id for ev in e.evidence)}]"
                if e.evidence else "[no evidence]"
            )
            bounds = (
                f", bounds={e.published_bounds}"
                if e.published_bounds is not None else ""
            )
            src = f", source={e.source_type}" if e.source_type else ""
            context_parts.append(
                f"  {e.parameter}={e.value:.3f} (conf {e.confidence:.2f}{src}{bounds}) "
                f"{cite} — {e.basis or 'no basis'}"
            )

    context = "\n".join(context_parts)

    try:
        audit = llm.complete(
            system_prompt=ANALYZE_PROMPT,
            user_message=context,
            temperature=0.4,  # slightly creative for prose
        )
        return audit.strip()
    except Exception as e:
        logger.warning(f"Failed to generate audit: {e}")
        return (
            f"**Adoption: {sim_result.aware_adoption*100:.0f}%** "
            f"(confidence band: {confidence_band.low*100:.0f}%–{confidence_band.high*100:.0f}%)\n\n"
            f"The simulation completed but the behavioral audit could not be generated. "
            f"See the force decomposition and sensitivity analysis above for details."
        )
