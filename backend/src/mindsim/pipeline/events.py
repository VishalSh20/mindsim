"""Stage 4e: EVENT INTERPRETER — full-context event processing.

Takes event text + full SimulationResult context, calls LLM for
per-force adjustments, applies adjustments, reruns simulation.
"""

from __future__ import annotations

import logging

import numpy as np

from mindsim.llm.client import LLMClient
from mindsim.llm.prompts import EVENT_PROMPT
from mindsim.models.config import SimulationConfig
from mindsim.models.results import EventForceAdjustment, EventResult, SimulationResult
from mindsim.pipeline.simulate import simulate

logger = logging.getLogger(__name__)


def process_event(
    event_text: str,
    sim_result: SimulationResult,
    config: SimulationConfig,
    llm: LLMClient,
    rng: np.random.Generator | None = None,
) -> tuple[SimulationResult, EventResult]:
    """Process a market event and rerun simulation.

    Args:
        event_text: Natural language description of the event.
        sim_result: Current simulation results (for context).
        config: Current SimulationConfig.
        llm: LLM client.
        rng: Random number generator.

    Returns:
        (new_sim_result, event_result) tuple.
    """
    if rng is None:
        rng = np.random.default_rng()

    logger.info(f"Processing event: {event_text}")

    # Build full context for the LLM
    context = _build_event_context(event_text, sim_result, config)

    # Get force adjustments from LLM
    data = llm.complete_json(
        system_prompt=EVENT_PROMPT,
        user_message=context,
        temperature=0.2,
    )

    # Apply adjustments to config
    modified_config = config.model_copy(deep=True)
    force_adjustments = []

    for adj in data.get("force_adjustments", []):
        force_name = adj.get("force", "")
        param_changes = adj.get("param_changes", {})
        magnitude = adj.get("magnitude", 0.0)
        mechanism = adj.get("mechanism", "")

        # Apply param changes
        for param_name, new_value in param_changes.items():
            cparam = getattr(modified_config.simulation_params, param_name, None)
            if cparam is not None and hasattr(cparam, "value"):
                cparam.value = float(new_value)
            elif param_name == "price":
                modified_config.simulation_params.price = float(new_value)
            elif param_name == "reference_price":
                modified_config.simulation_params.reference_price.value = float(new_value)

        force_adjustments.append(EventForceAdjustment(
            force=force_name,
            magnitude=float(magnitude),
            mechanism=mechanism,
            new_value=param_changes.get("reference_price"),
        ))

    # Rerun simulation with modified params
    new_result = simulate(
        config=modified_config,
        n_agents=sim_result.n_agents,
        rng=rng,
    )

    # Build event result — all user-facing metrics use total_adoption
    event_result = EventResult(
        event_text=event_text,
        adoption_before=sim_result.total_adoption,
        adoption_after=new_result.total_adoption,
        adoption_delta=new_result.total_adoption - sim_result.total_adoption,
        force_adjustments=force_adjustments,
        segment_effects=data.get("segment_effects", {}),
        second_order_effects=data.get("second_order_effects", []),
    )

    new_result.event_results.append(event_result)

    logger.info(
        f"Event processed: adoption {sim_result.total_adoption*100:.1f}% → "
        f"{new_result.total_adoption*100:.1f}% "
        f"({event_result.adoption_delta*100:+.1f}pp)"
    )

    return new_result, event_result


def _build_event_context(
    event_text: str,
    sim_result: SimulationResult,
    config: SimulationConfig,
) -> str:
    """Build full context for event interpretation."""
    forces = sim_result.force_decomposition.as_dict()
    params = config.simulation_params

    awareness_pct = sim_result.n_aware / max(sim_result.n_agents, 1) * 100
    parts = [
        f"# MARKET EVENT",
        f'"{event_text}"',
        "",
        f"# CURRENT SIMULATION STATE",
        f"Total adoption: {sim_result.total_adoption*100:.1f}% of all agents",
        f"Aware adoption: {sim_result.aware_adoption*100:.1f}% of aware agents",
        f"Awareness: {sim_result.n_aware}/{sim_result.n_agents} ({awareness_pct:.0f}%)",
        "",
        "## Current Force Decomposition (convertible pool averages)",
    ]

    for name, val in forces.items():
        parts.append(f"  {name}: {val:+.4f}")

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
            parts.append(
                f"  {seg.bracket}: {seg.adoption_rate*100:.1f}%"
            )

    return "\n".join(parts)
