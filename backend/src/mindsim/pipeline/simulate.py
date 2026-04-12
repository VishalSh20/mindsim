"""Stage 4: SIMULATE — orchestrate the simulation engine.

Generates population → awareness filter → compute forces → decisions → segment.
Pure math, 0 LLM calls, <100ms for 1000 agents.
"""

from __future__ import annotations

import logging

import numpy as np

from mindsim.engine.archetypes import ArchetypeSet, load_archetypes
from mindsim.engine.forces import compute_decisions, compute_forces, compute_weighted_forces
from mindsim.engine.population import AGENT_DTYPE, apply_awareness, generate_population
from mindsim.models.config import SimulationConfig
from mindsim.models.results import (
    ArchetypeSegment,
    ForceDecomposition,
    IncomeSegment,
    ProximitySegment,
    SimulationResult,
)

logger = logging.getLogger(__name__)


def simulate(
    config: SimulationConfig,
    n_agents: int = 1000,
    rng: np.random.Generator | None = None,
    archetype_set: ArchetypeSet | None = None,
) -> SimulationResult:
    """Run the full simulation pipeline.

    Args:
        config: SimulationConfig from the calibrate stage.
        n_agents: Number of agents to simulate.
        rng: Random number generator for reproducibility.
        archetype_set: Archetype definitions. Loads from YAML if None.

    Returns:
        SimulationResult with adoption rates, force decomposition, segments.
    """
    if rng is None:
        rng = np.random.default_rng()
    if archetype_set is None:
        archetype_set = load_archetypes()

    params = config.simulation_params
    archetype_names = archetype_set.names

    # 4a. GENERATE POPULATION
    logger.info(f"Generating {n_agents} agents...")
    agents = generate_population(
        n=n_agents,
        population_config=config.population_config,
        archetype_set=archetype_set,
        sim_params=params,
        rng=rng,
    )

    # 4b. AWARENESS FILTER
    awareness_dict = {
        "innovator": params.awareness.innovator,
        "early_adopter": params.awareness.early_adopter,
        "early_majority": params.awareness.early_majority,
        "late_majority": params.awareness.late_majority,
        "laggard": params.awareness.laggard,
    }
    agents = apply_awareness(
        agents, awareness_dict, archetype_names,
        rng=rng, archetype_set=archetype_set,
    )

    n_aware = int(agents["aware"].sum())
    logger.info(f"Aware agents: {n_aware}/{n_agents} ({n_aware/n_agents*100:.1f}%)")

    # 4c. COMPUTE FORCES
    forces = compute_forces(agents, params)

    # 4d. DECISIONS + SEGMENTATION
    adopt_prob, decisions = compute_decisions(forces, temperature=3.0, rng=rng)

    # Overall adoption
    total_adoption = float(decisions.sum() / n_agents) if n_agents > 0 else 0.0
    aware_adoption = float(decisions[agents["aware"]].sum() / max(n_aware, 1))

    # Boundary-weighted force decomposition
    weighted = compute_weighted_forces(forces, adopt_prob)
    force_decomposition = ForceDecomposition(
        prospect_value=weighted.get("prospect_value", 0.0),
        anchoring=weighted.get("anchoring", 0.0),
        status_quo=weighted.get("status_quo", 0.0),
        social_proof=weighted.get("social_proof", 0.0),
        fomo=weighted.get("fomo", 0.0),
        hyperbolic_discounting=weighted.get("hyperbolic_discounting", 0.0),
        identity_signaling=weighted.get("identity_signaling", 0.0),
    )

    # Segment by archetype
    by_archetype = []
    for i, aname in enumerate(archetype_names):
        mask = agents["archetype_id"] == i
        total = int(mask.sum())
        if total == 0:
            by_archetype.append(ArchetypeSegment(
                name=aname, adoption_rate=0.0, count=0, total=0
            ))
            continue
        adopted = int(decisions[mask].sum())
        by_archetype.append(ArchetypeSegment(
            name=aname,
            adoption_rate=adopted / total,
            count=adopted,
            total=total,
        ))

    # Segment by income (terciles)
    by_income = _segment_by_income(agents, decisions)

    # Segment by proximity
    by_proximity = _segment_by_proximity(adopt_prob, agents["aware"])

    result = SimulationResult(
        total_adoption=total_adoption,
        aware_adoption=aware_adoption,
        n_agents=n_agents,
        n_aware=n_aware,
        force_decomposition=force_decomposition,
        by_archetype=by_archetype,
        by_income=by_income,
        by_proximity=by_proximity,
    )

    # Store raw data for interactive deep dives
    result._agent_forces = forces
    result._agent_probs = adopt_prob
    result._agent_decisions = decisions
    result._agents = agents

    logger.info(
        f"Simulation complete: {total_adoption*100:.1f}% total adoption, "
        f"{aware_adoption*100:.1f}% aware adoption"
    )

    return result


def _segment_by_income(
    agents: np.ndarray,
    decisions: np.ndarray,
) -> list[IncomeSegment]:
    """Segment adoption by income terciles."""
    income = agents["income"]
    p33 = np.percentile(income, 33)
    p66 = np.percentile(income, 66)

    segments = []
    for bracket, low, high in [
        ("low", 0, p33),
        ("mid", p33, p66),
        ("high", p66, float("inf")),
    ]:
        mask = (income >= low) & (income < high) if high != float("inf") else (income >= low)
        if bracket == "high":
            mask = income >= low
        total = int(mask.sum())
        if total == 0:
            segments.append(IncomeSegment(bracket=bracket, adoption_rate=0.0, count=0, total=0))
            continue
        adopted = int(decisions[mask].sum())
        segments.append(IncomeSegment(
            bracket=bracket,
            adoption_rate=adopted / total,
            count=adopted,
            total=total,
        ))

    return segments


def _segment_by_proximity(
    adopt_prob: np.ndarray,
    aware_mask: np.ndarray,
) -> ProximitySegment:
    """Segment agents by decision proximity (locked/convertible/unreachable)."""
    n = len(adopt_prob)
    if n == 0:
        return ProximitySegment()

    aware = aware_mask.sum()
    if aware == 0:
        return ProximitySegment(locked=0, convertible=0, unreachable=1.0)

    aware_probs = adopt_prob[aware_mask]

    locked = float((aware_probs > 0.7).sum() / aware)
    convertible = float(((aware_probs >= 0.3) & (aware_probs <= 0.7)).sum() / aware)
    unreachable = float((aware_probs < 0.3).sum() / aware)

    return ProximitySegment(
        locked=locked,
        convertible=convertible,
        unreachable=unreachable,
    )
