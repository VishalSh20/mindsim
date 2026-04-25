"""Wave 8 — KPI dashboard extraction tests.

No LLM. Pure-Python checks against constructed `SimulationResult`
fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest

from mindsim.models.config import (
    CalibratedParam,
    PopulationConfig,
    ReferencePriceParam,
    SimulationConfig,
    SimulationParams,
)
from mindsim.models.results import (
    ForceDecomposition,
    RoundSnapshot,
    SensitivityResult,
    SimulationResult,
)
from mindsim.pipeline.kpi import (
    CONVERTIBLE_HIGH,
    CONVERTIBLE_LOW,
    build_kpi_dashboard,
)
from mindsim.pipeline.simulate import simulate


def _config():
    return SimulationConfig(
        simulation_params=SimulationParams(
            price=20.0,
            reference_price=ReferencePriceParam(value=18.0, basis="r"),
            category_penetration=CalibratedParam(value=0.30),
            category_growth=CalibratedParam(value=0.5),
            perceived_benefit=CalibratedParam(value=0.7),
            benefit_certainty=CalibratedParam(value=0.6),
            switching_cost=CalibratedParam(value=0.4),
            social_visibility=CalibratedParam(value=0.5),
            identity_signal=CalibratedParam(value=0.3),
            present_bias_beta=CalibratedParam(value=0.75),
            fomo_intensity=CalibratedParam(value=0.4),
            product_adoption_rate=CalibratedParam(value=0.05),
            requires_behavior_change=CalibratedParam(value=0.3),
            feature_matrix=[],
        ),
        population_config=PopulationConfig(),
    )


def _seed_run(seed: int = 1, n_agents: int = 400, n_rounds: int = 4):
    rng = np.random.default_rng(seed)
    config = _config()
    result = simulate(config, n_agents=n_agents, rng=rng, n_rounds=n_rounds)
    return result


# ───────────────────── adoption summary ─────────────────────


class TestAdoptionSummary:
    def test_time_to_50pct_when_threshold_crossed(self):
        result = SimulationResult(
            total_adoption=0.6,
            n_agents=100,
            rounds=[
                RoundSnapshot(round=1, total_adoption=0.10, aware_count=20),
                RoundSnapshot(round=2, total_adoption=0.30, aware_count=40),
                RoundSnapshot(round=3, total_adoption=0.55, aware_count=60),
                RoundSnapshot(round=4, total_adoption=0.60, aware_count=70),
            ],
        )
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        assert kpi.adoption.time_to_50pct == 3

    def test_time_to_50pct_none_when_never_reached(self):
        result = SimulationResult(
            total_adoption=0.30,
            rounds=[
                RoundSnapshot(round=1, total_adoption=0.10),
                RoundSnapshot(round=2, total_adoption=0.30),
            ],
        )
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        assert kpi.adoption.time_to_50pct is None

    def test_chasm_round_detects_growth_stall_past_innovator_share(self):
        result = SimulationResult(
            total_adoption=0.25,
            rounds=[
                RoundSnapshot(round=1, total_adoption=0.05),  # below innovator
                RoundSnapshot(round=2, total_adoption=0.10),  # below innovator
                RoundSnapshot(round=3, total_adoption=0.20),  # crossed 16%
                RoundSnapshot(round=4, total_adoption=0.205),  # stall (+0.5pp)
                RoundSnapshot(round=5, total_adoption=0.25),
            ],
        )
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        assert kpi.adoption.chasm_round == 4

    def test_chasm_round_none_when_growth_continues(self):
        result = SimulationResult(
            total_adoption=0.55,
            rounds=[
                RoundSnapshot(round=i, total_adoption=0.10 * i)
                for i in range(1, 6)
            ],
        )
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        assert kpi.adoption.chasm_round is None


# ───────────────────── force dominance ─────────────────────


class TestForceDominance:
    def test_top_driver_is_largest_force(self):
        result = SimulationResult(
            total_adoption=0.5,
            force_decomposition=ForceDecomposition(
                prospect_value=0.40,
                anchoring=-0.10,
                status_quo=-0.30,
                social_proof=0.20,
            ),
        )
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        assert kpi.force_dominance.top_driver == "prospect_value"
        assert kpi.force_dominance.top_blocker == "status_quo"

    def test_segment_variance_uses_per_archetype_means(self):
        result = _seed_run()
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        # Variance dict should be populated since simulate() attaches forces.
        assert kpi.force_dominance.segment_variance
        # Every variance entry is non-negative.
        assert all(v >= 0 for v in kpi.force_dominance.segment_variance.values())


# ───────────────────── convertible pool ─────────────────────


class TestConvertiblePool:
    def test_counts_agents_in_window(self):
        # Hand-craft probs so we can count exactly.
        probs = np.array([0.10, 0.42, 0.55, 0.61, 0.95])
        result = SimulationResult(total_adoption=0.5)
        result._agent_probs = probs
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        # Two values fall in [0.40, 0.60]: 0.42 and 0.55.
        assert kpi.convertible_pool == 2

    def test_returns_zero_when_no_probs_attached(self):
        result = SimulationResult(total_adoption=0.5)
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        assert kpi.convertible_pool == 0


# ───────────────────── cascade ─────────────────────


class TestCascade:
    def test_first_cluster_to_critical_mass(self):
        result = SimulationResult(
            total_adoption=0.4,
            rounds=[
                RoundSnapshot(round=1, cluster_adoption={0: 0.1, 1: 0.05}),
                RoundSnapshot(round=2, cluster_adoption={0: 0.55, 1: 0.10}),
                RoundSnapshot(round=3, cluster_adoption={0: 0.70, 1: 0.50}),
            ],
        )
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        assert kpi.cascade is not None
        assert kpi.cascade.first_cluster_crossed_critical_mass == 0
        assert kpi.cascade.cluster_spread_rounds == {0: 2, 1: 3}

    def test_no_cascade_when_no_cluster_data(self):
        result = SimulationResult(
            total_adoption=0.4,
            rounds=[RoundSnapshot(round=1, total_adoption=0.4)],
        )
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        assert kpi.cascade is None


# ───────────────────── sensitivity ─────────────────────


class TestTopSensitivityParams:
    def test_top_3_by_swing(self):
        result = SimulationResult(total_adoption=0.5)
        sens = [
            SensitivityResult(
                parameter="alpha", base_adoption=0.5,
                low_adoption=0.4, high_adoption=0.6, swing=20.0, confidence=0.5,
            ),
            SensitivityResult(
                parameter="beta", base_adoption=0.5,
                low_adoption=0.45, high_adoption=0.55, swing=10.0, confidence=0.5,
            ),
            SensitivityResult(
                parameter="gamma", base_adoption=0.5,
                low_adoption=0.30, high_adoption=0.70, swing=40.0, confidence=0.5,
            ),
            SensitivityResult(
                parameter="delta", base_adoption=0.5,
                low_adoption=0.48, high_adoption=0.52, swing=4.0, confidence=0.5,
            ),
        ]
        kpi = build_kpi_dashboard(result, sensitivity_results=sens)
        assert kpi.top_sensitivity_params == ["gamma", "alpha", "beta"]

    def test_empty_sensitivity_returns_empty_list(self):
        result = SimulationResult(total_adoption=0.5)
        kpi = build_kpi_dashboard(result, sensitivity_results=[])
        assert kpi.top_sensitivity_params == []


# ───────────────────── end-to-end via simulate ─────────────────────


class TestEndToEnd:
    def test_full_dashboard_populates_on_simulated_run(self):
        result = _seed_run()
        kpi = build_kpi_dashboard(result, sensitivity_results=[], validation_score=0.85)
        # Adoption, force dominance, convertible pool all populated.
        assert kpi.adoption.total == pytest.approx(result.total_adoption)
        assert kpi.force_dominance.top_driver
        assert kpi.force_dominance.top_blocker
        assert kpi.convertible_pool >= 0
        assert kpi.validation_score == 0.85
