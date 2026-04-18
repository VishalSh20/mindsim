"""Tests for the Phase enum and agent-state stamping (Wave 3 Commit 1).

No behavior change — existing single-shot simulations should produce the
same results. These tests verify:
  - AGENT_DTYPE has the new state fields
  - Initial state: everyone UNAWARE, zero tenure/investment
  - apply_awareness advances newly-aware agents to AWARE with strength 0.5
  - `aware` boolean still mirrors (phase != UNAWARE)
"""
import numpy as np
import pytest

from mindsim.engine.population import (
    AGENT_DTYPE,
    apply_awareness,
    generate_population,
)
from mindsim.models.config import ARCHETYPE_NAMES
from mindsim.models.state import (
    ADOPTED_PHASES,
    AWARE_PHASES,
    DECISION_PHASES,
    Phase,
    phase_name,
)


class TestPhaseEnum:
    def test_phase_values_ordered(self):
        assert Phase.UNAWARE < Phase.AWARE < Phase.CONSIDERING
        assert Phase.CONSIDERING < Phase.TRIALING < Phase.ADOPTED

    def test_decision_phases_excludes_terminals(self):
        assert Phase.ADOPTED.value not in DECISION_PHASES
        assert Phase.LOCKED_IN.value not in DECISION_PHASES
        assert Phase.CHURNED.value not in DECISION_PHASES
        assert Phase.UNAWARE.value not in DECISION_PHASES

    def test_decision_phases_includes_decision_making(self):
        for p in (Phase.AWARE, Phase.CONSIDERING, Phase.TRIALING):
            assert p.value in DECISION_PHASES

    def test_aware_phases_excludes_unaware(self):
        assert Phase.UNAWARE.value not in AWARE_PHASES
        # Churned agents remember — still aware.
        assert Phase.CHURNED.value in AWARE_PHASES

    def test_phase_name_roundtrip(self):
        for p in Phase:
            assert phase_name(p.value) == p.name.lower()

    def test_phase_name_unknown(self):
        assert phase_name(99).startswith("unknown")


class TestAgentDtypeExtensions:
    def test_dtype_has_phase_field(self):
        assert "phase" in AGENT_DTYPE.names
        assert AGENT_DTYPE.fields["phase"][0] == np.dtype(np.int8)

    def test_dtype_has_all_state_fields(self):
        expected = {
            "phase", "awareness_strength", "tenure_current_solution",
            "investment_depth", "trial_outcome", "trial_rounds_remaining",
            "cluster_id",
        }
        assert expected.issubset(set(AGENT_DTYPE.names))


class TestInitialStateStamping:
    def test_initial_phase_is_unaware_before_awareness(self):
        """Before apply_awareness is called, everyone is UNAWARE."""
        rng = np.random.default_rng(42)
        agents = generate_population(n=100, rng=rng)
        # In generate_population, aware defaults to True but phase=UNAWARE.
        # The simulate stage is what calls apply_awareness afterwards.
        assert (agents["phase"] == Phase.UNAWARE).all()

    def test_initial_awareness_strength_zero(self):
        rng = np.random.default_rng(42)
        agents = generate_population(n=100, rng=rng)
        assert (agents["awareness_strength"] == 0.0).all()

    def test_initial_tenure_and_investment_zero(self):
        rng = np.random.default_rng(42)
        agents = generate_population(n=100, rng=rng)
        assert (agents["tenure_current_solution"] == 0.0).all()
        assert (agents["investment_depth"] == 0.0).all()

    def test_initial_trial_fields(self):
        rng = np.random.default_rng(42)
        agents = generate_population(n=100, rng=rng)
        assert (agents["trial_outcome"] == -1).all()
        assert (agents["trial_rounds_remaining"] == 0).all()

    def test_initial_cluster_id_zero(self):
        rng = np.random.default_rng(42)
        agents = generate_population(n=100, rng=rng)
        assert (agents["cluster_id"] == 0).all()


class TestApplyAwarenessPromotesPhase:
    def test_aware_agents_get_phase_aware(self):
        rng = np.random.default_rng(42)
        agents = generate_population(n=500, rng=rng)
        # Full awareness — everyone becomes aware
        awareness = {name: 1.0 for name in ARCHETYPE_NAMES}
        agents = apply_awareness(
            agents, awareness, ARCHETYPE_NAMES, rng=rng
        )
        assert (agents["phase"] == Phase.AWARE).all()
        assert (agents["awareness_strength"] == 0.5).all()
        assert agents["aware"].all()

    def test_unaware_agents_stay_unaware(self):
        rng = np.random.default_rng(42)
        agents = generate_population(n=500, rng=rng)
        awareness = {name: 0.0 for name in ARCHETYPE_NAMES}
        agents = apply_awareness(
            agents, awareness, ARCHETYPE_NAMES, rng=rng
        )
        assert (agents["phase"] == Phase.UNAWARE).all()
        assert (agents["awareness_strength"] == 0.0).all()
        assert not agents["aware"].any()

    def test_mixed_awareness_splits_phase_correctly(self):
        rng = np.random.default_rng(42)
        agents = generate_population(n=1000, rng=rng)
        awareness = {
            "innovator": 0.9,
            "early_adopter": 0.6,
            "early_majority": 0.3,
            "late_majority": 0.1,
            "laggard": 0.02,
        }
        agents = apply_awareness(
            agents, awareness, ARCHETYPE_NAMES, rng=rng
        )
        # aware boolean must match phase != UNAWARE
        expected_aware = agents["phase"] != Phase.UNAWARE
        assert np.array_equal(agents["aware"], expected_aware)

    def test_aware_agents_have_strength_05(self):
        """Newly-aware agents start with a moderate awareness_strength."""
        rng = np.random.default_rng(0)
        agents = generate_population(n=1000, rng=rng)
        awareness = {name: 0.5 for name in ARCHETYPE_NAMES}
        agents = apply_awareness(
            agents, awareness, ARCHETYPE_NAMES, rng=rng
        )
        aware_mask = agents["phase"] == Phase.AWARE
        unaware_mask = agents["phase"] == Phase.UNAWARE
        assert (agents["awareness_strength"][aware_mask] == 0.5).all()
        assert (agents["awareness_strength"][unaware_mask] == 0.0).all()


class TestBackwardCompatNoBehaviorChange:
    """The existing single-shot simulate() must still produce sane output."""

    def test_existing_pipeline_still_runs(self):
        from mindsim.models.config import SimulationConfig
        from mindsim.pipeline.simulate import simulate
        config = SimulationConfig()
        # Feature matrix stays empty → v1 scalar path → status quo heavy.
        result = simulate(config=config, n_agents=200, rng=np.random.default_rng(1))
        assert 0.0 <= result.total_adoption <= 1.0
        assert result.n_agents == 200
