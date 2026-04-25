"""Stage 4: SIMULATE — orchestrate the simulation engine.

v2-middle Wave 3: multi-round simulation.

Per MECHANICS-v2 §"The Key Insight": no round is special. The same
force computation + state transitions run every round. The S-curve,
the chasm, and the adoption cascade all emerge from the mechanics
rather than being hardcoded.

Each round:
  1. decay awareness (UNAWARE drifters + AWARE decay)
  2. promote UNAWARE → AWARE (marketing + word-of-mouth reach)
  3. promote AWARE → CONSIDERING (strength threshold + relevance gate)
  4. compute forces (only DECISION_PHASES agents)
  5. draw CONSIDERING → ADOPTED via logistic
  6. advance ADOPTED → LOCKED_IN or CHURNED

Still pure NumPy, still <1s for N=8 rounds × 1000 agents.
"""

from __future__ import annotations

import logging

import numpy as np

from mindsim.engine.archetypes import ArchetypeSet, load_archetypes
from mindsim.engine.clusters import compute_cluster_adoption_rates
from mindsim.engine.competitor_tiers import apply_tiered_reference_prices
from mindsim.engine.forces import compute_decisions, compute_forces, compute_weighted_forces
from mindsim.engine.population import AGENT_DTYPE, apply_awareness, generate_population
from mindsim.engine.state_machine import (
    advance_adopted,
    advance_trialing,
    decay_awareness,
    decide_considering_to_adopted,
    phase_counts,
    promote_aware_to_considering,
    promote_unaware_to_aware,
    sync_aware_flag,
    total_adoption_count,
)
from mindsim.models.config import SimulationConfig
from mindsim.models.results import (
    ArchetypeSegment,
    ForceDecomposition,
    IncomeSegment,
    ProximitySegment,
    RoundSnapshot,
    SimulationResult,
)
from mindsim.models.state import ADOPTED_PHASES, Phase

logger = logging.getLogger(__name__)


DEFAULT_N_ROUNDS = 8


def simulate(
    config: SimulationConfig,
    n_agents: int = 1000,
    rng: np.random.Generator | None = None,
    archetype_set: ArchetypeSet | None = None,
    n_rounds: int | None = None,
    market: "MarketContext | None" = None,
) -> SimulationResult:
    """Run the full simulation.

    Args:
        config: SimulationConfig from the calibrate stage.
        n_agents: Number of agents to simulate.
        rng: Random number generator for reproducibility.
        archetype_set: Archetype definitions. Loads from YAML if None.
        n_rounds: How many rounds to simulate. `None` uses config's
            `simulation_params.n_rounds` (default 8). Pass 1 for the
            old v1 single-shot semantics.

    Returns:
        SimulationResult with per-round snapshots, adoption rates,
        force decomposition, and segments.
    """
    if rng is None:
        rng = np.random.default_rng()
    if archetype_set is None:
        archetype_set = load_archetypes()

    params = config.simulation_params
    archetype_names = archetype_set.names
    if n_rounds is None:
        n_rounds = int(getattr(params, "n_rounds", DEFAULT_N_ROUNDS) or DEFAULT_N_ROUNDS)
    n_rounds = max(1, n_rounds)

    # 4a. GENERATE POPULATION
    logger.info("Generating %d agents…", n_agents)
    agents = generate_population(
        n=n_agents,
        population_config=config.population_config,
        archetype_set=archetype_set,
        sim_params=params,
        rng=rng,
    )

    # v2-middle Wave 5: per-agent tiered reference price override.
    # No-op when the market context is None or no competitor carries a
    # `market_share` value (keeps the archetype-based stamping from
    # population.py as the fallback).
    if market is not None and market.verified_competitors:
        category_default = float(params.reference_price.value)
        apply_tiered_reference_prices(
            agents=agents,
            competitors=market.verified_competitors,
            category_default_price=category_default,
            rng=rng,
        )

    # 4b. INITIAL AWARENESS
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
    sync_aware_flag(agents)

    # 4c. MULTI-ROUND LOOP
    rounds_log, final_forces, final_adopt_prob = run_rounds(
        agents, params, n_rounds=n_rounds, rng=rng, starting_round=1
    )

    # 4d. POST-SIMULATION SUMMARY
    result = summarize_state(
        agents=agents,
        n_agents=n_agents,
        archetype_names=archetype_names,
        final_forces=final_forces,
        final_adopt_prob=final_adopt_prob,
        rounds_log=rounds_log,
    )

    logger.info(
        "Simulation complete (%d rounds): %.1f%% total, %.1f%% of aware",
        n_rounds,
        result.total_adoption * 100,
        result.aware_adoption * 100,
    )
    return result


def run_rounds(
    agents: np.ndarray,
    params,
    n_rounds: int,
    rng: np.random.Generator,
    starting_round: int = 1,
) -> tuple[list[RoundSnapshot], dict[str, np.ndarray], np.ndarray]:
    """Execute `n_rounds` of the per-round phase / force / decision loop.

    Mutates `agents` in place. Returns the per-round snapshot log and the
    final-round force / probability arrays so callers can build (or update)
    a SimulationResult.

    Used by `simulate()` for fresh runs and by Wave 6's
    `pipeline/events.py` to apply additional rounds onto a persisted
    population without regenerating it.
    """
    n_agents = len(agents)
    rounds_log: list[RoundSnapshot] = []
    final_forces: dict[str, np.ndarray] | None = None
    final_adopt_prob: np.ndarray | None = None

    for offset in range(n_rounds):
        round_idx = starting_round + offset

        # Cluster-local adoption rates (Wave 4) — computed once at the top
        # of each round from the state carried over from the previous
        # round. Used by both the WOM awareness promotion and the
        # cluster-local social-proof force.
        cluster_rates = compute_cluster_adoption_rates(agents)

        decay_awareness(agents)
        promote_unaware_to_aware(
            agents,
            product_adoption_rate=float(params.product_adoption_rate.value),
            rng=rng,
            cluster_rates=cluster_rates,
        )
        promote_aware_to_considering(agents, params, rng=rng)
        sync_aware_flag(agents)

        forces = compute_forces(agents, params, cluster_rates=cluster_rates)
        adopt_prob, _ = compute_decisions(forces, temperature=3.0, rng=rng)

        decide_considering_to_adopted(
            agents,
            adopt_prob,
            rng=rng,
            free_trial_active=bool(getattr(params, "free_trial_active", False)),
            free_trial_duration_rounds=int(
                getattr(params, "free_trial_duration_rounds", 3)
            ),
            free_trial_entry_threshold=float(
                getattr(params, "free_trial_entry_threshold", 0.20)
            ),
        )
        advance_trialing(agents, rng=rng)
        n_locked, n_churned = advance_adopted(agents, rng=rng)
        sync_aware_flag(agents)

        total_adopt = total_adoption_count(agents)
        post_cluster_rates = compute_cluster_adoption_rates(agents)
        rounds_log.append(
            RoundSnapshot(
                round=round_idx,
                phase_counts=phase_counts(agents),
                cluster_adoption={
                    i: float(post_cluster_rates[i])
                    for i in range(len(post_cluster_rates))
                },
                total_adoption=total_adopt / max(n_agents, 1),
                aware_count=int(agents["aware"].sum()),
            )
        )
        logger.debug(
            "round %d: adopted=%d locked_in=%d churned=%d (+%d locked, +%d churned)",
            round_idx,
            int((agents["phase"] == int(Phase.ADOPTED)).sum()),
            int((agents["phase"] == int(Phase.LOCKED_IN)).sum()),
            int((agents["phase"] == int(Phase.CHURNED)).sum()),
            n_locked,
            n_churned,
        )

        final_forces = forces
        final_adopt_prob = adopt_prob

    assert final_forces is not None and final_adopt_prob is not None
    return rounds_log, final_forces, final_adopt_prob


def snapshot_state(
    agents: np.ndarray,
    params,
    rng: np.random.Generator,
    archetype_names: list[str] | None = None,
) -> SimulationResult:
    """Snapshot a loaded agent array into a SimulationResult without
    advancing any rounds.

    Computes a single force pass + decision-probability draw so the
    resulting `SimulationResult` carries the same `_agent_forces` /
    `_agent_probs` invariants as a freshly-simulated one. Used by
    Wave 6's session loader so events can be applied without first
    re-running the multi-round simulation.
    """
    if archetype_names is None:
        archetype_names = list(load_archetypes().names)
    sync_aware_flag(agents)
    cluster_rates = compute_cluster_adoption_rates(agents)
    forces = compute_forces(agents, params, cluster_rates=cluster_rates)
    adopt_prob, _ = compute_decisions(forces, temperature=3.0, rng=rng)
    return summarize_state(
        agents=agents,
        n_agents=len(agents),
        archetype_names=archetype_names,
        final_forces=forces,
        final_adopt_prob=adopt_prob,
        rounds_log=[],
    )


def summarize_state(
    agents: np.ndarray,
    n_agents: int,
    archetype_names: list[str],
    final_forces: dict[str, np.ndarray],
    final_adopt_prob: np.ndarray,
    rounds_log: list[RoundSnapshot],
) -> SimulationResult:
    """Build a SimulationResult from a finished agent array + final-round arrays.

    Pure post-processing — no transitions, no RNG. Safe to call repeatedly
    on the same agents array (e.g. after each event applied on a
    persisted population).
    """
    phase_arr = agents["phase"]
    decisions = np.isin(phase_arr, np.array(ADOPTED_PHASES, dtype=phase_arr.dtype))
    n_decided = int(decisions.sum())

    total_adoption = float(n_decided / n_agents) if n_agents > 0 else 0.0
    aware_like = agents["aware"] | decisions
    n_aware_like = int(aware_like.sum())
    aware_adoption = float(
        decisions[aware_like].sum() / max(n_aware_like, 1)
    ) if n_aware_like > 0 else 0.0

    weighted = compute_weighted_forces(final_forces, final_adopt_prob)
    force_decomposition = ForceDecomposition(
        prospect_value=weighted.get("prospect_value", 0.0),
        anchoring=weighted.get("anchoring", 0.0),
        status_quo=weighted.get("status_quo", 0.0),
        social_proof=weighted.get("social_proof", 0.0),
        fomo=weighted.get("fomo", 0.0),
        hyperbolic_discounting=weighted.get("hyperbolic_discounting", 0.0),
        identity_signaling=weighted.get("identity_signaling", 0.0),
    )

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

    by_income = _segment_by_income(agents, decisions)
    by_proximity = _segment_by_proximity(final_adopt_prob, agents["aware"])

    result = SimulationResult(
        total_adoption=total_adoption,
        aware_adoption=aware_adoption,
        n_agents=n_agents,
        n_aware=n_aware_like,
        force_decomposition=force_decomposition,
        by_archetype=by_archetype,
        by_income=by_income,
        by_proximity=by_proximity,
        rounds=rounds_log,
    )

    result._agent_forces = final_forces
    result._agent_probs = final_adopt_prob
    result._agent_decisions = decisions
    result._agents = agents
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
