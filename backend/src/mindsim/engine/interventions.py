"""Intervention ranking — candidate interventions + rerun simulation.

For each candidate intervention, modifies specific params, reruns the
simulation, and ranks by adoption lift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mindsim.engine.forces import compute_decisions, compute_forces
from mindsim.models.config import SimulationConfig, SimulationParams
from mindsim.models.results import InterventionResult


@dataclass
class InterventionSpec:
    """Definition of a candidate intervention."""

    name: str
    description: str
    mechanism: str
    modifications: dict[str, float]  # param_name -> multiplier or new value
    mode: str = "multiply"  # "multiply" or "set"


# Standard interventions per ARCHITECTURE.md Stage 5c
STANDARD_INTERVENTIONS: list[InterventionSpec] = [
    InterventionSpec(
        name="Free trial",
        description="Offer a free trial to eliminate upfront cost",
        mechanism="Removes the immediate price loss entirely (price→$0 during trial), breaking the loss_aversion × discounting multiplicative penalty. Also boosts benefit_certainty (users experience value before paying) and activates Cialdini's commitment/consistency.",
        modifications={"price_zero": True, "benefit_certainty": 1.4},
        mode="multiply",
    ),
    InterventionSpec(
        name="Price cut 20%",
        description="Reduce price by 20%",
        mechanism="Directly reduces the loss component of prospect value. Most impactful for price-sensitive segments.",
        modifications={"price_factor": 0.8},
        mode="set",
    ),
    InterventionSpec(
        name="Annual discount",
        description="Offer annual billing with ~30% savings",
        mechanism="Exploits hyperbolic discounting — annual commitment reduces per-month perceived cost and locks in the behavior change.",
        modifications={"present_bias_beta": 1.15},
        mode="multiply",
    ),
    InterventionSpec(
        name="Social proof push",
        description="Boost social proof via testimonials, usage stats, endorsements",
        mechanism="Doubles the effective social proof signal. Most impactful when benefit_certainty is low.",
        modifications={"product_adoption_rate": 3.0},
        mode="multiply",
    ),
    InterventionSpec(
        name="Freemium tier",
        description="Add a free tier to shift reference price",
        mechanism="Shifts reference price down and removes loss aversion for entry. Also serves as awareness booster.",
        modifications={"benefit_certainty": 1.2, "switching_cost": 0.6},
        mode="multiply",
    ),
]


def rank_interventions(
    agents: np.ndarray,
    config: SimulationConfig,
    base_adoption: float,
    interventions: list[InterventionSpec] | None = None,
    rng: np.random.Generator | None = None,
) -> list[InterventionResult]:
    """Run each candidate intervention and rank by adoption lift.

    Args:
        agents: Agent array (reused — no regeneration).
        config: Current SimulationConfig.
        base_adoption: Current adoption rate.
        interventions: List of interventions to test. Defaults to STANDARD_INTERVENTIONS.
        rng: Random number generator.

    Returns:
        List of InterventionResult sorted by lift (descending).
    """
    if interventions is None:
        interventions = STANDARD_INTERVENTIONS
    if rng is None:
        rng = np.random.default_rng(42)

    results = []

    for intervention in interventions:
        modified_params = config.simulation_params.model_copy(deep=True)

        # Apply modifications
        for param_name, factor in intervention.modifications.items():
            # Special-case: price modifications
            if param_name == "price_zero":
                modified_params.price = 0.0
                continue
            if param_name == "price_factor":
                modified_params.price *= factor
                continue

            cparam = getattr(modified_params, param_name, None)
            if cparam is None:
                continue
            if intervention.mode == "multiply":
                new_val = cparam.value * factor
                # Clamp probability-bounded params (those already in 0-1 range)
                if 0.0 <= cparam.value <= 1.0:
                    new_val = min(new_val, 1.0)
                cparam.value = new_val
            else:
                cparam.value = factor

        forces = compute_forces(agents, modified_params)
        _, decisions = compute_decisions(forces, rng=rng)

        aware_mask = agents["aware"]
        if aware_mask.sum() == 0:
            new_adoption = 0.0
        else:
            new_adoption = float(decisions[aware_mask].sum() / aware_mask.sum())

        lift = (new_adoption - base_adoption) * 100.0  # percentage points

        results.append(InterventionResult(
            name=intervention.name,
            description=intervention.description,
            adoption_before=base_adoption,
            adoption_after=new_adoption,
            lift_pp=lift,
            mechanism=intervention.mechanism,
        ))

    # Sort by lift (biggest first)
    results.sort(key=lambda r: r.lift_pp, reverse=True)
    return results
