"""Stage 4e: EVENT INTERPRETER — mutate persisted population in place.

Wave 6 rewrite. Earlier waves regenerated the entire population on
every event, so an agent who adopted last round had no memory the next.
This version mutates the existing agent array, optionally re-stamps
per-agent product params if the event changed any, then runs a small
number of additional rounds via the same state-machine `run_rounds`
helper that powers `simulate()`.

Effects of this change:
  - Adopted / LOCKED_IN agents stay decided unless the state-machine
    churns them out (small per-round probability, weighted by
    investment_depth and conscientiousness — see advance_adopted).
  - Considering / Aware agents see the new params next round and may
    flip in either direction.
  - Event order matters: a price cut after most agents have already
    locked in produces a smaller lift than the same cut delivered early.

The LLM call (`EVENT_PROMPT`) is unchanged in intent — emit per-force
adjustments — but receives a richer per-segment trajectory snapshot so
it can reason about adopters vs considerers separately.
"""

from __future__ import annotations

import logging

import numpy as np

from mindsim.engine.archetypes import load_archetypes
from mindsim.engine.population import (
    PARAM_TO_AGENT_FIELD,
    REF_PRICE_AGENT_FIELD,
    restamp_agent_param,
)
from mindsim.llm.client import LLMClient
from mindsim.llm.prompts import EVENT_PROMPT
from mindsim.models.config import SimulationConfig
from mindsim.models.results import EventForceAdjustment, EventResult, SimulationResult
from mindsim.models.state import Phase
from mindsim.pipeline.simulate import (
    DEFAULT_N_ROUNDS,
    run_rounds,
    summarize_state,
)

logger = logging.getLogger(__name__)


# Sub-rounds the population is advanced by after each event applies.
# Two is enough to let aware agents re-cross the consideration threshold
# (CONSIDERATION_STRENGTH_GAIN = 0.20 per round) without giving every
# agent a fresh adoption draw.
EVENT_SUBROUNDS = 2

# Param fields the engine has per-agent stamps for; if an event changes
# any of these we re-stamp the affected agents.
RESTAMPABLE_PARAMS = set(PARAM_TO_AGENT_FIELD.keys()) | {"reference_price"}


def process_event(
    event_text: str,
    sim_result: SimulationResult,
    config: SimulationConfig,
    llm: LLMClient,
    rng: np.random.Generator | None = None,
    n_subrounds: int = EVENT_SUBROUNDS,
) -> tuple[SimulationResult, EventResult]:
    """Apply an event onto the population carried by `sim_result`.

    The agents array on `sim_result._agents` is mutated in place. A new
    SimulationResult is returned that summarises the post-event state
    while preserving the prior `event_results` history.

    Args:
        event_text: natural-language event description.
        sim_result: result from a prior `simulate()` (or a prior event).
            Must carry `_agents` (true for any in-process result; loaded
            sessions wire this up via `events_from_session`).
        config: SimulationConfig — modified in place with the event's
            param changes so subsequent rounds see them.
        llm: LLM client.
        rng: optional generator. Created fresh if None — for
            reproducibility, callers should pass the run's RNG.
        n_subrounds: how many additional rounds to advance the
            population after applying the event. 2 = enough to let
            consideration thresholds re-trigger without flooding agents
            with adoption draws.

    Returns:
        (new_sim_result, event_result). `new_sim_result._agents` is the
        same mutated array; `event_result.adoption_delta` is signed.
    """
    if rng is None:
        rng = np.random.default_rng()

    agents = sim_result._agents
    if agents is None:
        raise ValueError(
            "process_event requires sim_result with attached agent array. "
            "Run simulate() first or load a session with attach_session()."
        )

    logger.info("Processing event: %s", event_text)
    adoption_before = sim_result.total_adoption

    # 1. Ask the LLM to interpret the event into per-force adjustments.
    context = _build_event_context(event_text, sim_result, config)
    data = llm.complete_json(
        system_prompt=EVENT_PROMPT,
        user_message=context,
        temperature=0.2,
    )

    # 2. Apply param changes to the (mutable) config and re-stamp any
    #    per-agent fields that depend on them.
    force_adjustments: list[EventForceAdjustment] = []
    changed_params: set[str] = set()

    for adj in data.get("force_adjustments", []) or []:
        force_name = adj.get("force", "")
        param_changes = adj.get("param_changes", {}) or {}
        magnitude = adj.get("magnitude", 0.0)
        mechanism = adj.get("mechanism", "")

        for param_name, new_value in param_changes.items():
            if not _apply_param_change(config, param_name, new_value):
                continue
            changed_params.add(param_name)

        force_adjustments.append(
            EventForceAdjustment(
                force=force_name,
                magnitude=float(magnitude),
                mechanism=mechanism,
                new_value=param_changes.get("reference_price"),
            )
        )

    for param_name in changed_params & RESTAMPABLE_PARAMS:
        cparam = (
            config.simulation_params.reference_price
            if param_name == "reference_price"
            else getattr(config.simulation_params, param_name, None)
        )
        if cparam is not None:
            restamp_agent_param(agents, param_name, cparam, rng)

    # 3. Advance the population n_subrounds further with the new params.
    archetype_set = load_archetypes()
    archetype_names = archetype_set.names
    starting_round = (sim_result.rounds[-1].round + 1) if sim_result.rounds else 1

    new_rounds, final_forces, final_adopt_prob = run_rounds(
        agents=agents,
        params=config.simulation_params,
        n_rounds=max(1, n_subrounds),
        rng=rng,
        starting_round=starting_round,
    )

    # 4. Build a fresh SimulationResult covering the post-event state.
    combined_rounds = list(sim_result.rounds) + new_rounds
    new_result = summarize_state(
        agents=agents,
        n_agents=sim_result.n_agents,
        archetype_names=list(archetype_names),
        final_forces=final_forces,
        final_adopt_prob=final_adopt_prob,
        rounds_log=combined_rounds,
    )
    new_result.event_results = list(sim_result.event_results)

    event_result = EventResult(
        event_text=event_text,
        adoption_before=adoption_before,
        adoption_after=new_result.total_adoption,
        adoption_delta=new_result.total_adoption - adoption_before,
        force_adjustments=force_adjustments,
        segment_effects=data.get("segment_effects", {}) or {},
        second_order_effects=data.get("second_order_effects", []) or [],
    )
    new_result.event_results.append(event_result)

    logger.info(
        "Event processed: adoption %.1f%% → %.1f%% (%+.1fpp)",
        adoption_before * 100,
        new_result.total_adoption * 100,
        event_result.adoption_delta * 100,
    )
    return new_result, event_result


def attach_session(
    sim_result: SimulationResult,
    agents: np.ndarray,
) -> SimulationResult:
    """Wire a session-loaded agent array onto a freshly-built result.

    Helper for the CLI's `--session` path: after `load_session()` we
    don't have a prior SimulationResult, so we build a thin one from the
    agents and let `process_event` mutate from there.
    """
    sim_result._agents = agents
    return sim_result


# ──────────────────────────── internals ────────────────────────────


def _apply_param_change(
    config: SimulationConfig,
    param_name: str,
    new_value: float,
) -> bool:
    """Apply a single LLM-emitted param change to the config in place.

    Returns True if anything was actually written.
    """
    try:
        new_value_f = float(new_value)
    except (TypeError, ValueError):
        logger.debug("event ignored non-numeric value for %s: %r", param_name, new_value)
        return False

    if param_name == "price":
        config.simulation_params.price = new_value_f
        return True
    if param_name == "reference_price":
        config.simulation_params.reference_price.value = new_value_f
        return True
    cparam = getattr(config.simulation_params, param_name, None)
    if cparam is not None and hasattr(cparam, "value"):
        cparam.value = new_value_f
        return True
    logger.debug("event referenced unknown param %s", param_name)
    return False


def _build_event_context(
    event_text: str,
    sim_result: SimulationResult,
    config: SimulationConfig,
) -> str:
    """Build the user-message payload for EVENT_PROMPT.

    Includes the per-segment trajectory summary the Wave 6 prompt expects:
    phase counts plus per-archetype phase breakdown so the LLM can reason
    about adopters / considerers separately.
    """
    forces = sim_result.force_decomposition.as_dict()
    params = config.simulation_params
    agents = sim_result._agents

    awareness_pct = sim_result.n_aware / max(sim_result.n_agents, 1) * 100
    parts = [
        "# MARKET EVENT",
        f'"{event_text}"',
        "",
        "# CURRENT SIMULATION STATE",
        f"Total adoption: {sim_result.total_adoption*100:.1f}% of all agents",
        f"Aware adoption: {sim_result.aware_adoption*100:.1f}% of aware agents",
        f"Awareness: {sim_result.n_aware}/{sim_result.n_agents} ({awareness_pct:.0f}%)",
        "",
        "## Current Force Decomposition (convertible pool averages)",
    ]
    for name, val in forces.items():
        parts.append(f"  {name}: {val:+.4f}")

    if agents is not None:
        parts.extend(["", "## Phase Distribution (agents currently in each phase)"])
        for p in Phase:
            n = int((agents["phase"] == int(p)).sum())
            if n == 0:
                continue
            parts.append(f"  {p.name.lower()}: {n}")

        parts.extend(["", "## Per-Archetype Phase Breakdown"])
        archetype_names = ["innovator", "early_adopter", "early_majority", "late_majority", "laggard"]
        for i, aname in enumerate(archetype_names):
            mask = agents["archetype_id"] == i
            total = int(mask.sum())
            if total == 0:
                continue
            adopted = int(np.isin(
                agents["phase"][mask],
                np.array([int(Phase.ADOPTED), int(Phase.LOCKED_IN)], dtype=agents["phase"].dtype),
            ).sum())
            considering = int((agents["phase"][mask] == int(Phase.CONSIDERING)).sum())
            aware = int((agents["phase"][mask] == int(Phase.AWARE)).sum())
            churned = int((agents["phase"][mask] == int(Phase.CHURNED)).sum())
            parts.append(
                f"  {aname}: adopted={adopted} considering={considering} "
                f"aware={aware} churned={churned} (n={total})"
            )

    parts.extend([
        "",
        "## Current Parameters",
        f"  Price: ${params.price:.2f}/mo",
        f"  Reference price: ${params.reference_price.value:.2f}",
        f"  Category penetration: {params.category_penetration.value:.2f}",
        f"  Benefit certainty: {params.benefit_certainty.value:.2f}",
        f"  Switching cost: {params.switching_cost.value:.2f}",
        f"  Social visibility: {params.social_visibility.value:.2f}",
        f"  Identity signal: {params.identity_signal.value:.2f}",
        f"  Category growth: {params.category_growth.value:.2f}",
        f"  Present bias β: {params.present_bias_beta.value:.2f}",
        f"  FOMO intensity: {params.fomo_intensity.value:.2f}",
        "",
        "## Segment Data",
    ])

    top_barrier = min(forces.items(), key=lambda x: x[1])
    top_driver = max(forces.items(), key=lambda x: x[1])
    parts.append(f"  Top barrier: {top_barrier[0]} ({top_barrier[1]:+.3f})")
    parts.append(f"  Top driver: {top_driver[0]} ({top_driver[1]:+.3f})")
    parts.append(f"  Convertible pool: {sim_result.by_proximity.convertible*100:.0f}%")
    parts.append(f"  Locked in: {sim_result.by_proximity.locked*100:.0f}%")
    parts.append(f"  Unreachable: {sim_result.by_proximity.unreachable*100:.0f}%")

    if sim_result.by_income:
        parts.append("")
        parts.append("## Income Segments")
        for seg in sim_result.by_income:
            parts.append(f"  {seg.bracket}: {seg.adoption_rate*100:.1f}%")

    parts.extend([
        "",
        "## Note for the model",
        "Already-ADOPTED and LOCKED_IN agents will not be re-drawn — your",
        "force adjustments influence agents in AWARE/CONSIDERING phases",
        "directly, and may produce churn for ADOPTED agents only via the",
        "state-machine churn check (typically small).",
    ])

    return "\n".join(parts)
