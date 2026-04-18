"""Unit tests for engine/state_machine.py transitions.

Every transition is tested in isolation against a hand-built agent array.
No LLM, no full pipeline.
"""
import numpy as np
import pytest

from mindsim.engine.population import AGENT_DTYPE
from mindsim.engine.state_machine import (
    BASE_CHURN_RATE,
    CONSIDERATION_STRENGTH_GAIN,
    INITIAL_AWARENESS_STRENGTH,
    INVESTMENT_GROWTH_PER_ROUND,
    LOCK_IN_THRESHOLD,
    advance_adopted,
    decay_awareness,
    decide_considering_to_adopted,
    phase_counts,
    promote_aware_to_considering,
    promote_unaware_to_aware,
    sync_aware_flag,
    total_adoption_count,
)
from mindsim.models.config import CalibratedParam, SimulationParams
from mindsim.models.state import Phase


def _blank_agents(n: int, phase: Phase = Phase.UNAWARE) -> np.ndarray:
    agents = np.zeros(n, dtype=AGENT_DTYPE)
    agents["phase"] = int(phase)
    agents["conscientiousness"] = 0.5
    # Feature weights set to uniform so relevance gate doesn't trip.
    for cat in ("core_value", "social_signal",
                "ongoing_cost", "switching_friction_reducer"):
        agents[f"feature_weight_{cat}"] = 0.25
    return agents


def _params_with_feature_matrix():
    """Params with one strong positive feature so relevance gate opens."""
    from mindsim.models.product import Feature
    params = SimulationParams(
        feature_matrix=[
            Feature(
                name="core", score=0.8, polarity="positive",
                category="core_value", certainty=0.8, visibility=0.3,
                time_to_value_months=0.5,
            ),
        ],
        consideration_threshold=0.4,
    )
    return params


class TestSyncAwareFlag:
    def test_syncs_unaware_to_false(self):
        a = _blank_agents(10, Phase.UNAWARE)
        sync_aware_flag(a)
        assert not a["aware"].any()

    def test_syncs_aware_considering_trialing_to_true(self):
        for p in (Phase.AWARE, Phase.CONSIDERING, Phase.TRIALING):
            a = _blank_agents(5, p)
            sync_aware_flag(a)
            assert a["aware"].all()

    def test_adopted_and_locked_in_are_not_aware_for_forces(self):
        """Adopted/locked agents already decided — not in the decision pool."""
        for p in (Phase.ADOPTED, Phase.LOCKED_IN, Phase.CHURNED):
            a = _blank_agents(5, p)
            sync_aware_flag(a)
            assert not a["aware"].any()


class TestDecayAwareness:
    def test_decays_aware_agents(self):
        a = _blank_agents(20, Phase.AWARE)
        a["awareness_strength"] = 0.8
        decay_awareness(a, decay=0.1, threshold=0.05)
        assert np.allclose(a["awareness_strength"], 0.72)
        assert (a["phase"] == int(Phase.AWARE)).all()

    def test_below_threshold_reverts_to_unaware(self):
        a = _blank_agents(5, Phase.AWARE)
        a["awareness_strength"] = 0.04
        decay_awareness(a, decay=0.1, threshold=0.05)
        assert (a["phase"] == int(Phase.UNAWARE)).all()
        assert (a["awareness_strength"] == 0.0).all()

    def test_does_not_touch_adopted(self):
        a = _blank_agents(5, Phase.ADOPTED)
        a["awareness_strength"] = 0.8
        decay_awareness(a)
        assert (a["awareness_strength"] == 0.8).all()


class TestPromoteUnawareToAware:
    def test_high_reach_promotes_most(self):
        a = _blank_agents(1000, Phase.UNAWARE)
        rng = np.random.default_rng(0)
        n = promote_unaware_to_aware(
            a, product_adoption_rate=0.0, rng=rng, marketing_reach=0.5
        )
        # ~50% become aware.
        assert 0.4 < n / 1000 < 0.6

    def test_no_unaware_no_op(self):
        a = _blank_agents(10, Phase.AWARE)
        rng = np.random.default_rng(0)
        n = promote_unaware_to_aware(a, product_adoption_rate=0.5, rng=rng)
        assert n == 0

    def test_promoted_get_initial_strength(self):
        a = _blank_agents(100, Phase.UNAWARE)
        rng = np.random.default_rng(0)
        promote_unaware_to_aware(a, product_adoption_rate=0.0, rng=rng, marketing_reach=1.0)
        assert (a["phase"] == int(Phase.AWARE)).all()
        assert np.allclose(a["awareness_strength"], INITIAL_AWARENESS_STRENGTH)

    def test_word_of_mouth_effect(self):
        """Higher product_adoption_rate → more newly-aware."""
        rng = np.random.default_rng(1)
        a_low = _blank_agents(500, Phase.UNAWARE)
        promote_unaware_to_aware(a_low, product_adoption_rate=0.0, rng=rng, marketing_reach=0.05)
        n_low = int((a_low["phase"] == int(Phase.AWARE)).sum())

        rng = np.random.default_rng(1)
        a_high = _blank_agents(500, Phase.UNAWARE)
        promote_unaware_to_aware(a_high, product_adoption_rate=0.5, rng=rng, marketing_reach=0.05)
        n_high = int((a_high["phase"] == int(Phase.AWARE)).sum())

        assert n_high > n_low


class TestPromoteAwareToConsidering:
    def test_strength_below_threshold_stays_aware(self):
        a = _blank_agents(10, Phase.AWARE)
        a["awareness_strength"] = 0.0
        params = _params_with_feature_matrix()
        params.consideration_threshold = 0.9
        rng = np.random.default_rng(0)
        n = promote_aware_to_considering(a, params, rng)
        # Gain of 0.2 + starting 0.0 = 0.2 < 0.9 threshold → none consider.
        assert n == 0
        assert (a["phase"] == int(Phase.AWARE)).all()

    def test_strength_above_threshold_promotes(self):
        a = _blank_agents(50, Phase.AWARE)
        a["awareness_strength"] = 0.4  # already above a 0.3 threshold
        params = _params_with_feature_matrix()
        params.consideration_threshold = 0.3
        rng = np.random.default_rng(0)
        n = promote_aware_to_considering(a, params, rng)
        assert n == 50
        assert (a["phase"] == int(Phase.CONSIDERING)).all()

    def test_repeated_exposure_adds_strength(self):
        a = _blank_agents(5, Phase.AWARE)
        a["awareness_strength"] = 0.1
        params = _params_with_feature_matrix()
        params.consideration_threshold = 0.9  # high enough none cross
        rng = np.random.default_rng(0)
        promote_aware_to_considering(a, params, rng)
        expected = 0.1 + CONSIDERATION_STRENGTH_GAIN
        assert np.allclose(a["awareness_strength"], expected)


class TestDecideConsideringToAdopted:
    def test_high_prob_leads_to_adoption(self):
        a = _blank_agents(100, Phase.CONSIDERING)
        probs = np.ones(100) * 0.99
        rng = np.random.default_rng(0)
        n_adopt, n_revert = decide_considering_to_adopted(a, probs, rng)
        assert n_adopt > 90
        # Count matches the phase distribution (adopted agents scattered).
        assert int((a["phase"] == int(Phase.ADOPTED)).sum()) == n_adopt

    def test_zero_prob_reverts_all(self):
        a = _blank_agents(50, Phase.CONSIDERING)
        a["awareness_strength"] = 0.5
        probs = np.zeros(50)
        rng = np.random.default_rng(0)
        n_adopt, n_revert = decide_considering_to_adopted(a, probs, rng)
        assert n_adopt == 0
        assert n_revert == 50
        assert (a["phase"] == int(Phase.AWARE)).all()

    def test_revert_penalises_awareness_strength(self):
        a = _blank_agents(10, Phase.CONSIDERING)
        a["awareness_strength"] = 0.6
        probs = np.zeros(10)
        rng = np.random.default_rng(0)
        decide_considering_to_adopted(a, probs, rng, revert_strength_penalty=0.2)
        assert np.allclose(a["awareness_strength"], 0.4)

    def test_adopters_get_initial_investment_depth(self):
        a = _blank_agents(20, Phase.CONSIDERING)
        probs = np.ones(20)
        rng = np.random.default_rng(0)
        decide_considering_to_adopted(a, probs, rng)
        assert (a["investment_depth"] == 0.10).all()


class TestAdvanceAdopted:
    def test_investment_grows(self):
        a = _blank_agents(10, Phase.ADOPTED)
        a["investment_depth"] = 0.2
        rng = np.random.default_rng(0)
        advance_adopted(a, rng, growth=0.1, base_churn=0.0)
        assert np.allclose(
            a["investment_depth"][a["phase"] == int(Phase.ADOPTED)], 0.3
        )

    def test_lock_in_when_threshold_crossed(self):
        a = _blank_agents(10, Phase.ADOPTED)
        a["investment_depth"] = 0.65
        rng = np.random.default_rng(0)
        n_locked, _ = advance_adopted(
            a, rng, growth=0.1, lock_in_threshold=0.7, base_churn=0.0
        )
        assert n_locked == 10
        assert (a["phase"] == int(Phase.LOCKED_IN)).all()

    def test_churn_small_for_invested_agents(self):
        a = _blank_agents(1000, Phase.ADOPTED)
        a["investment_depth"] = 0.6
        a["conscientiousness"] = 0.5
        rng = np.random.default_rng(0)
        _, n_churn = advance_adopted(a, rng, growth=0.0, base_churn=0.05)
        # (1 - 0.6) × (1 - 0.25) × 0.05 = 0.015 → ~15 out of 1000
        assert n_churn < 50

    def test_zero_depth_agents_churn_more(self):
        a_low = _blank_agents(1000, Phase.ADOPTED)
        a_low["investment_depth"] = 0.0
        a_low["conscientiousness"] = 0.0
        rng = np.random.default_rng(0)
        _, n_churn_low = advance_adopted(a_low, rng, growth=0.0, base_churn=0.10)

        a_high = _blank_agents(1000, Phase.ADOPTED)
        a_high["investment_depth"] = 0.5
        a_high["conscientiousness"] = 0.5
        rng = np.random.default_rng(0)
        _, n_churn_high = advance_adopted(a_high, rng, growth=0.0, base_churn=0.10)

        assert n_churn_low > n_churn_high


class TestSummaries:
    def test_phase_counts_all_phases_in_dict(self):
        a = _blank_agents(10, Phase.AWARE)
        pc = phase_counts(a)
        for p in Phase:
            assert p.name.lower() in pc

    def test_phase_counts_sum_to_n(self):
        a = _blank_agents(50, Phase.CONSIDERING)
        pc = phase_counts(a)
        assert sum(pc.values()) == 50

    def test_total_adoption_counts_adopted_plus_locked_in(self):
        a = np.zeros(20, dtype=AGENT_DTYPE)
        a["phase"][:5] = int(Phase.ADOPTED)
        a["phase"][5:10] = int(Phase.LOCKED_IN)
        a["phase"][10:15] = int(Phase.CHURNED)
        a["phase"][15:] = int(Phase.UNAWARE)
        assert total_adoption_count(a) == 10
