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

logger = logging.getLogger(__name__)


def analyze(
    sim_result: SimulationResult,
    config: SimulationConfig,
    llm: LLMClient,
    rng: np.random.Generator | None = None,
) -> AnalysisReport:
    """Run full analysis: sensitivity + confidence + interventions + audit.

    Args:
        sim_result: Results from the simulate stage.
        config: SimulationConfig with parameter provenance.
        llm: LLM client for behavioral audit.
        rng: Random number generator.

    Returns:
        Complete AnalysisReport.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    agents = sim_result._agents
    if agents is None:
        raise ValueError("SimulationResult missing agent data for analysis")

    # 5a. SENSITIVITY ANALYSIS
    # All user-facing metrics use total_adoption (adoption among ALL agents)
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

    # 5d. BEHAVIORAL AUDIT (LLM)
    logger.info("Generating behavioral audit...")
    audit_text = _generate_audit(
        sim_result=sim_result,
        config=config,
        sensitivity=sensitivity,
        confidence_band=confidence_band,
        interventions=interventions,
        llm=llm,
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
        context_parts.append("## Interventions")
        for intv in interventions:
            context_parts.append(
                f"  {intv.name}: {intv.lift_pp:+.0f}pp "
                f"({intv.adoption_before*100:.0f}%→{intv.adoption_after*100:.0f}%)"
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
