"""Unit tests for engine/feature_forces.py (Wave 2 Commit 3).

Tests the per-feature gain/loss aggregation in isolation, plus an
integration test that two products with identical aggregate strength
but different feature profiles produce different segment-level adoption.
"""
import numpy as np

from mindsim.engine.feature_forces import (
    compute_feature_gain,
    compute_feature_loss,
    compute_weighted_certainty,
    compute_weighted_time_to_value_months,
    compute_weighted_visibility,
)
from mindsim.engine.population import generate_population
from mindsim.engine.forces import compute_decisions, compute_forces
from mindsim.models.config import (
    CalibratedParam,
    PopulationConfig,
    ReferencePriceParam,
    SimulationParams,
)
from mindsim.models.product import Feature


def _mkpop(n: int = 500, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return generate_population(n=n, rng=rng)


def _core(score: float = 0.5, cert: float = 0.5) -> Feature:
    return Feature(
        name="core", score=score, polarity="positive", category="core_value",
        certainty=cert, visibility=0.3, time_to_value_months=1.0,
    )


def _cost(score: float = 0.5, cert: float = 0.7) -> Feature:
    return Feature(
        name="cost", score=score, polarity="negative", category="ongoing_cost",
        certainty=cert, visibility=0.1, time_to_value_months=0.0,
    )


def _signal(score: float = 0.5) -> Feature:
    return Feature(
        name="sig", score=score, polarity="positive", category="social_signal",
        certainty=0.5, visibility=0.6, time_to_value_months=0.5,
    )


def _sfr(score: float = 0.5) -> Feature:
    return Feature(
        name="integ", score=score, polarity="positive",
        category="switching_friction_reducer",
        certainty=0.6, visibility=0.2, time_to_value_months=0.0,
    )


class TestComputeFeatureGain:
    def test_empty_matrix_returns_zeros(self):
        agents = _mkpop()
        gain = compute_feature_gain(agents, [])
        assert gain.shape == (len(agents),)
        assert (gain == 0).all()

    def test_only_negative_features_produces_zero_gain(self):
        agents = _mkpop()
        gain = compute_feature_gain(agents, [_cost(0.5)])
        assert (gain == 0).all()

    def test_gain_scales_with_score(self):
        agents = _mkpop()
        g_low = compute_feature_gain(agents, [_core(score=0.2)])
        g_high = compute_feature_gain(agents, [_core(score=0.8)])
        assert (g_high >= g_low).all()
        assert g_high.mean() > g_low.mean() + 0.01

    def test_gain_differs_across_archetypes(self):
        """Innovators weight core_value higher → higher gain on core-heavy product."""
        agents = _mkpop(n=2000, seed=7)
        gain = compute_feature_gain(agents, [_core(score=0.8)])
        inno_mean = gain[agents["archetype_id"] == 0].mean()
        lag_mean = gain[agents["archetype_id"] == 4].mean()
        assert inno_mean > lag_mean + 0.05

    def test_gain_accumulates_over_multiple_features(self):
        agents = _mkpop()
        one = compute_feature_gain(agents, [_core(0.6)])
        two = compute_feature_gain(agents, [_core(0.6), _signal(0.6)])
        # Second positive feature strictly increases gain for every agent.
        assert (two >= one).all()
        assert two.mean() > one.mean()


class TestComputeFeatureLoss:
    def test_only_negative_features_contribute(self):
        agents = _mkpop()
        loss_pos_only = compute_feature_loss(agents, [_core(0.9), _signal(0.9)])
        assert (loss_pos_only == 0).all()
        loss_mixed = compute_feature_loss(
            agents, [_core(0.9), _cost(score=0.7)]
        )
        assert (loss_mixed > 0).any()

    def test_laggards_weight_ongoing_cost_more(self):
        """Laggards have higher feature_weight_ongoing_cost → higher loss
        on cost-heavy products than innovators."""
        agents = _mkpop(n=2000, seed=9)
        loss = compute_feature_loss(agents, [_cost(score=0.8)])
        inno_mean = loss[agents["archetype_id"] == 0].mean()
        lag_mean = loss[agents["archetype_id"] == 4].mean()
        assert lag_mean > inno_mean + 0.1


class TestAggregateHelpers:
    def test_weighted_ttv_zero_for_empty_pos(self):
        agents = _mkpop()
        ttv = compute_weighted_time_to_value_months(agents, [_cost(0.5)])
        assert (ttv == 0).all()

    def test_weighted_ttv_matches_single_feature(self):
        agents = _mkpop()
        feat = _core()
        feat.time_to_value_months = 4.0
        ttv = compute_weighted_time_to_value_months(agents, [feat])
        assert np.allclose(ttv, 4.0)

    def test_weighted_visibility_fallback(self):
        agents = _mkpop()
        vis = compute_weighted_visibility(agents, [], fallback=0.25)
        assert (vis == 0.25).all()

    def test_weighted_visibility_respects_positive_only(self):
        agents = _mkpop()
        neg_only = [_cost(0.5)]
        vis = compute_weighted_visibility(agents, neg_only, fallback=0.1)
        assert (vis == 0.1).all()

    def test_weighted_certainty_with_fixture(self):
        agents = _mkpop()
        f1 = _core(cert=0.4)
        f2 = _signal(0.5)
        cert = compute_weighted_certainty(agents, [f1, f2])
        # Both features have cert=0.4 and 0.5; weighted mean lies between.
        assert (cert >= 0.4).all()
        assert (cert <= 0.6).all()


class TestFeatureMatrixIntegratedInForces:
    """End-to-end: forces.py using the feature-matrix path."""

    def _base_params(self, features: list[Feature], price: float = 20.0) -> SimulationParams:
        return SimulationParams(
            price=price,
            reference_price=ReferencePriceParam(value=25.0),
            feature_matrix=features,
            category_penetration=CalibratedParam(value=0.15),
            category_growth=CalibratedParam(value=0.5),
            benefit_certainty=CalibratedParam(value=0.5),
            perceived_benefit=CalibratedParam(value=0.5),
            time_to_value=CalibratedParam(value=0.2),
            requires_behavior_change=CalibratedParam(value=0.3),
            switching_cost=CalibratedParam(value=0.5),
            social_visibility=CalibratedParam(value=0.4),
            identity_signal=CalibratedParam(value=0.3),
            present_bias_beta=CalibratedParam(value=0.75),
            fomo_intensity=CalibratedParam(value=0.3),
            product_adoption_rate=CalibratedParam(value=0.1),
        )

    def test_empty_feature_matrix_falls_back_to_scalar(self):
        """With no features, forces still computes (v1 scalar path)."""
        agents = _mkpop(n=300, seed=5)
        params = self._base_params(features=[])
        forces = compute_forces(agents, params)
        # Sanity: aware agents produce finite prospect values.
        aware = agents["aware"]
        assert np.isfinite(forces["prospect_value"][aware]).all()

    def test_feature_matrix_produces_segment_differences(self):
        """Core-value-strong product benefits innovators more than laggards."""
        core_strong = [
            _core(score=0.9),
            _signal(score=0.3),
            _cost(score=0.3),
            _sfr(score=0.4),
        ]
        agents = _mkpop(n=2000, seed=11)
        params = self._base_params(features=core_strong)
        forces = compute_forces(agents, params)
        aware = agents["aware"]
        prospect = forces["prospect_value"]
        inno_mean = prospect[aware & (agents["archetype_id"] == 0)].mean()
        lag_mean = prospect[aware & (agents["archetype_id"] == 4)].mean()
        assert inno_mean > lag_mean

    def test_cost_heavy_product_hurts_laggards_more_than_innovators(self):
        """Laggards weight ongoing_cost heavily → a product with a strong
        negative cost feature should drop laggard prospect more than innovator's."""
        cost_heavy_neutral = [
            _core(score=0.6),
            _signal(score=0.3),
            _cost(score=0.9),  # strong ongoing cost
            _sfr(score=0.3),
        ]
        cost_light_neutral = [
            _core(score=0.6),
            _signal(score=0.3),
            _cost(score=0.1),  # weak ongoing cost
            _sfr(score=0.3),
        ]
        agents = _mkpop(n=2000, seed=13)
        p_heavy = self._base_params(features=cost_heavy_neutral)
        p_light = self._base_params(features=cost_light_neutral)

        f_heavy = compute_forces(agents, p_heavy)
        f_light = compute_forces(agents, p_light)
        aware = agents["aware"]

        lag = agents["archetype_id"] == 4
        inno = agents["archetype_id"] == 0

        lag_delta = (
            f_heavy["prospect_value"][aware & lag].mean()
            - f_light["prospect_value"][aware & lag].mean()
        )
        inno_delta = (
            f_heavy["prospect_value"][aware & inno].mean()
            - f_light["prospect_value"][aware & inno].mean()
        )
        # Both should drop (heavy cost hurts everyone), but laggards drop further.
        assert lag_delta < inno_delta

    def test_two_products_same_aggregate_different_profile_diverge(self):
        """Products with the same mean score but different feature mix
        produce materially different segment-level adoption."""
        # Product A: strong core_value, weak social_signal. Mean = 0.5.
        product_a = [
            _core(score=0.8),
            _signal(score=0.2),
            _cost(score=0.3),
            _sfr(score=0.4),
        ]
        # Product B: inverted.
        product_b = [
            _core(score=0.2),
            _signal(score=0.8),
            _cost(score=0.3),
            _sfr(score=0.4),
        ]

        agents = _mkpop(n=2000, seed=17)
        params_a = self._base_params(features=product_a)
        params_b = self._base_params(features=product_b)

        fa = compute_forces(agents, params_a)
        fb = compute_forces(agents, params_b)
        aware = agents["aware"]

        # Innovators weight social_signal 0.35 vs 0.50 core_value: they care
        # more about core than signal → product A (core-strong) beats product B.
        inno = agents["archetype_id"] == 0
        assert fa["prospect_value"][aware & inno].mean() > fb["prospect_value"][aware & inno].mean()
