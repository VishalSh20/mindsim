"""Sensitivity analysis — ±30% rerun on uncertain parameters.

Pure recomputation, no LLM calls. For every parameter with confidence
below a threshold (default 0.80), reruns the **full multi-round**
simulation at ±30% and records adoption swing in percentage points.

v2-middle Wave 3 change: pre-Wave-3 sensitivity ran a single-shot
compute_forces + compute_decisions. With the multi-round loop in place,
we now invoke `simulate()` directly per perturbation so the sensitivity
numbers reflect the same mechanics the main run uses (state transitions,
awareness decay, lock-in, churn).

Cost: ~12 params × 2 perturbations × 8 rounds ≈ 192 force computations,
plus 24 population regenerations. Typical run ~3-5 s for 1000 agents.
"""
from __future__ import annotations

import logging

import numpy as np

from mindsim.models.config import CalibratedParam, SimulationConfig, SimulationParams
from mindsim.models.results import SensitivityResult

logger = logging.getLogger(__name__)


# Parameters eligible for sensitivity analysis.
SENSITIVE_PARAMS = [
    "category_penetration",
    "benefit_certainty",
    "perceived_benefit",
    "time_to_value",
    "switching_cost",
    "social_visibility",
    "identity_signal",
    "present_bias_beta",
    "fomo_intensity",
    "category_growth",
    "product_adoption_rate",
]


def run_sensitivity_analysis(
    agents: np.ndarray | None,
    config: SimulationConfig,
    base_adoption: float,
    swing_fraction: float = 0.30,
    confidence_threshold: float = 0.80,
    rng: np.random.Generator | None = None,
    n_agents: int = 1000,
) -> list[SensitivityResult]:
    """Run sensitivity analysis on uncertain parameters over the full
    multi-round simulation.

    Args:
        agents: Unused in Wave 3 — retained for back-compat callers.
            Multi-round sensitivity regenerates agents each perturbation.
        config: Current SimulationConfig.
        base_adoption: Base adoption rate to compare against.
        swing_fraction: How much to perturb (0.30 = ±30%).
        confidence_threshold: Only test params below this confidence.
        rng: Random number generator for stable results.
        n_agents: Number of agents to simulate per perturbation.

    Returns:
        List of SensitivityResult, sorted by swing magnitude (descending).
    """
    # Import here to avoid a circular dependency between sensitivity and
    # simulate (both live in different packages but both reference forces).
    from mindsim.pipeline.simulate import simulate

    if rng is None:
        rng = np.random.default_rng(42)  # fixed seed for stable sensitivity

    results: list[SensitivityResult] = []
    params = config.simulation_params

    for param_name in SENSITIVE_PARAMS:
        cparam: CalibratedParam | None = getattr(params, param_name, None)
        if cparam is None:
            continue
        if cparam.confidence >= confidence_threshold:
            continue

        base_value = cparam.value
        low_value = base_value * (1.0 - swing_fraction)
        high_value = base_value * (1.0 + swing_fraction)

        low_adoption = _rerun_full_sim(
            config, param_name=param_name, new_value=low_value,
            rng=_child_rng(rng, f"{param_name}_lo"), n_agents=n_agents, simulate_fn=simulate,
        )
        high_adoption = _rerun_full_sim(
            config, param_name=param_name, new_value=high_value,
            rng=_child_rng(rng, f"{param_name}_hi"), n_agents=n_agents, simulate_fn=simulate,
        )

        swing_pp = abs(high_adoption - low_adoption) * 100.0
        results.append(SensitivityResult(
            parameter=param_name,
            base_adoption=base_adoption,
            low_adoption=low_adoption,
            high_adoption=high_adoption,
            swing=swing_pp,
            confidence=cparam.confidence,
        ))

    # Reference price (ReferencePriceParam — separate path)
    ref = params.reference_price
    if ref.value > 0 and ref.confidence < confidence_threshold:
        low_adoption = _rerun_full_sim(
            config, param_name="reference_price",
            new_value=ref.value * (1.0 - swing_fraction),
            rng=_child_rng(rng, "ref_lo"), n_agents=n_agents, simulate_fn=simulate,
        )
        high_adoption = _rerun_full_sim(
            config, param_name="reference_price",
            new_value=ref.value * (1.0 + swing_fraction),
            rng=_child_rng(rng, "ref_hi"), n_agents=n_agents, simulate_fn=simulate,
        )
        swing_pp = abs(high_adoption - low_adoption) * 100.0
        results.append(SensitivityResult(
            parameter="reference_price",
            base_adoption=base_adoption,
            low_adoption=low_adoption,
            high_adoption=high_adoption,
            swing=swing_pp,
            confidence=ref.confidence,
        ))

    results.sort(key=lambda r: r.swing, reverse=True)
    return results


def _rerun_full_sim(
    config: SimulationConfig,
    param_name: str,
    new_value: float,
    rng: np.random.Generator,
    n_agents: int,
    simulate_fn,
) -> float:
    """Run a full multi-round simulation with a single param overridden."""
    modified = config.model_copy(deep=True)
    if param_name == "reference_price":
        modified.simulation_params.reference_price.value = new_value
    else:
        cparam = getattr(modified.simulation_params, param_name)
        cparam.value = new_value

    result = simulate_fn(config=modified, n_agents=n_agents, rng=rng)
    return float(result.total_adoption)


def _child_rng(parent: np.random.Generator, tag: str) -> np.random.Generator:
    """Derive a deterministic child RNG from a parent + string tag.

    Ensures each perturbation uses its own stream but the overall run
    remains reproducible given the parent seed.
    """
    # Simple hash of tag into an int32 seed; combined with parent state.
    tag_hash = int(abs(hash(tag)) % (2**31))
    return np.random.default_rng(parent.integers(0, 2**31) ^ tag_hash)
