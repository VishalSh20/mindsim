"""Sensitivity analysis — ±30% rerun on uncertain parameters.

Pure recomputation, no LLM calls. For every parameter with
confidence < 0.8, reruns simulation at ±30% and records adoption swing.
"""

from __future__ import annotations

import numpy as np

from mindsim.engine.forces import compute_decisions, compute_forces
from mindsim.models.config import CalibratedParam, SimulationConfig, SimulationParams
from mindsim.models.results import SensitivityResult


# Parameters eligible for sensitivity analysis
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
    agents: np.ndarray,
    config: SimulationConfig,
    base_adoption: float,
    swing_fraction: float = 0.30,
    confidence_threshold: float = 0.80,
    rng: np.random.Generator | None = None,
) -> list[SensitivityResult]:
    """Run sensitivity analysis on uncertain parameters.

    For each param with confidence < threshold:
        1. Set param to value × (1 - swing)
        2. Rerun forces + decisions
        3. Record adoption rate (low)
        4. Set param to value × (1 + swing)
        5. Rerun forces + decisions
        6. Record adoption rate (high)
        7. swing = high - low (in percentage points)

    Args:
        agents: Agent array (pre-generated, reused).
        config: Current SimulationConfig.
        base_adoption: Base adoption rate to compare against.
        swing_fraction: How much to perturb (0.30 = ±30%).
        confidence_threshold: Only test params below this confidence.
        rng: Random number generator for stable results.

    Returns:
        List of SensitivityResult, sorted by swing magnitude (descending).
    """
    if rng is None:
        rng = np.random.default_rng(42)  # fixed seed for reproducible sensitivity

    results = []
    params = config.simulation_params

    for param_name in SENSITIVE_PARAMS:
        cparam: CalibratedParam = getattr(params, param_name, None)
        if cparam is None:
            continue
        if cparam.confidence >= confidence_threshold:
            continue

        base_value = cparam.value

        # --- Low scenario (param -30%) ---
        low_value = base_value * (1.0 - swing_fraction)
        low_adoption = _rerun_with_override(
            agents, params, param_name, low_value, rng
        )

        # --- High scenario (param +30%) ---
        high_value = base_value * (1.0 + swing_fraction)
        high_adoption = _rerun_with_override(
            agents, params, param_name, high_value, rng
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

    # Sort by swing magnitude (biggest uncertainty first)
    results.sort(key=lambda r: r.swing, reverse=True)
    return results


def _rerun_with_override(
    agents: np.ndarray,
    params: SimulationParams,
    param_name: str,
    new_value: float,
    rng: np.random.Generator,
) -> float:
    """Rerun simulation with a single param overridden.

    Creates a modified copy of params and runs forces + decisions.
    Returns adoption rate among aware agents.
    """
    # Deep copy params and override one field
    modified = params.model_copy(deep=True)
    cparam = getattr(modified, param_name)
    cparam.value = new_value

    forces = compute_forces(agents, modified)
    adopt_prob, decisions = compute_decisions(forces, rng=rng)

    aware_mask = agents["aware"]
    if aware_mask.sum() == 0:
        return 0.0

    return float(decisions[aware_mask].sum() / aware_mask.sum())
