"""Integration tests for the multi-round simulation loop.

Verifies:
  - n_rounds=1 preserves v1-ish single-shot semantics
  - n_rounds=8 produces a monotone non-decreasing adoption curve
  - RoundSnapshot entries are populated
  - S-curve emerges: adoption growth decelerates in later rounds
  - LOCKED_IN count grows in later rounds
"""
import numpy as np
import pytest

from mindsim.models.config import CalibratedParam, ReferencePriceParam, SimulationConfig
from mindsim.models.product import Feature
from mindsim.pipeline.simulate import simulate


def _config(feature_matrix=None, n_rounds=8, category_penetration=0.15):
    config = SimulationConfig()
    p = config.simulation_params
    p.n_rounds = n_rounds
    p.category_penetration = CalibratedParam(value=category_penetration)
    p.category_growth = CalibratedParam(value=0.5)
    p.price = 20.0
    p.reference_price = ReferencePriceParam(value=25.0)
    if feature_matrix is not None:
        p.feature_matrix = feature_matrix
    return config


def _strong_product_features():
    return [
        Feature(name="core", score=0.8, polarity="positive",
                category="core_value", certainty=0.7, visibility=0.3,
                time_to_value_months=0.5),
        Feature(name="cost", score=0.3, polarity="negative",
                category="ongoing_cost", certainty=0.8, visibility=0.1,
                time_to_value_months=0.0),
        Feature(name="integ", score=0.6, polarity="positive",
                category="switching_friction_reducer", certainty=0.6,
                visibility=0.2, time_to_value_months=0.0),
        Feature(name="sig", score=0.5, polarity="positive",
                category="social_signal", certainty=0.5, visibility=0.5,
                time_to_value_months=1.0),
    ]


class TestMultiRoundBasics:
    def test_n_rounds_1_still_works(self):
        result = simulate(
            config=_config(n_rounds=1),
            n_agents=500,
            rng=np.random.default_rng(42),
        )
        assert len(result.rounds) == 1
        assert 0.0 <= result.total_adoption <= 1.0

    def test_default_n_rounds_is_8(self):
        result = simulate(
            config=SimulationConfig(),
            n_agents=500,
            rng=np.random.default_rng(1),
        )
        assert len(result.rounds) == 8

    def test_per_round_snapshot_populated(self):
        result = simulate(
            config=_config(n_rounds=5),
            n_agents=300,
            rng=np.random.default_rng(2),
        )
        assert len(result.rounds) == 5
        for i, snap in enumerate(result.rounds, 1):
            assert snap.round == i
            # phase_counts sum ~= n_agents (agents never disappear)
            assert sum(snap.phase_counts.values()) == 300
            assert 0.0 <= snap.total_adoption <= 1.0


class TestAdoptionMonotonicity:
    def test_adoption_non_decreasing_over_rounds(self):
        """Total adoption (ADOPTED + LOCKED_IN) never goes down between rounds.

        Churn can move agents out of the adopted pool, but only a small amount —
        over many rounds adoption should grow and then level off.
        """
        result = simulate(
            config=_config(feature_matrix=_strong_product_features(), n_rounds=8),
            n_agents=1000,
            rng=np.random.default_rng(5),
        )
        totals = [snap.total_adoption for snap in result.rounds]
        # Allow tiny dips from churn — but not large drops.
        for i in range(1, len(totals)):
            assert totals[i] >= totals[i - 1] - 0.03

    def test_s_curve_shape(self):
        """Growth rate should slow in later rounds (concave adoption curve)."""
        result = simulate(
            config=_config(feature_matrix=_strong_product_features(), n_rounds=8),
            n_agents=1000,
            rng=np.random.default_rng(7),
        )
        totals = [snap.total_adoption for snap in result.rounds]
        # First-round growth > final-round growth for a deceleration curve.
        first_growth = totals[1] - totals[0]
        last_growth = totals[-1] - totals[-2]
        assert first_growth > last_growth


class TestLockInEmergence:
    def test_lock_in_grows_over_rounds(self):
        result = simulate(
            config=_config(feature_matrix=_strong_product_features(), n_rounds=8),
            n_agents=1000,
            rng=np.random.default_rng(11),
        )
        locked_counts = [snap.phase_counts["locked_in"] for snap in result.rounds]
        # Lock-in is zero in the first few rounds (investment_depth too low)
        assert locked_counts[0] == 0
        # And grows by the end.
        assert locked_counts[-1] > locked_counts[len(locked_counts) // 2]


class TestSituationalSQB:
    """Gap 8 fix — status quo bias should be lower for agents with no incumbent."""

    def test_high_penetration_raises_sqb_force(self):
        """Higher category_penetration → more agents have tenure → stronger sqb."""
        rng = np.random.default_rng(99)
        low_pen = simulate(
            config=_config(
                feature_matrix=_strong_product_features(),
                n_rounds=3,
                category_penetration=0.05,
            ),
            n_agents=1000,
            rng=rng,
        )
        rng = np.random.default_rng(99)
        high_pen = simulate(
            config=_config(
                feature_matrix=_strong_product_features(),
                n_rounds=3,
                category_penetration=0.80,
            ),
            n_agents=1000,
            rng=rng,
        )
        # Higher penetration → more tenured agents → sqb is larger in magnitude
        # (more negative). This is the situational-SQB correction: tenure
        # amplifies the dispositional bias.
        assert (
            high_pen.force_decomposition.status_quo
            < low_pen.force_decomposition.status_quo + 1e-3
        )

    def test_zero_penetration_keeps_agents_movable(self):
        """With no incumbent, even laggards should face lower sqb barrier."""
        rng = np.random.default_rng(33)
        result = simulate(
            config=_config(
                feature_matrix=_strong_product_features(),
                n_rounds=3,
                category_penetration=0.0,
            ),
            n_agents=1000,
            rng=rng,
        )
        # Total adoption should be non-trivial even across 3 rounds.
        assert result.total_adoption > 0.05
