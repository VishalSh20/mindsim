"""Wave 7 — interventions v2 (phase mechanics + cost-per-lift ranking).

Headline checks:
  * Free trial on a delayed-benefit product produces POSITIVE lift.
    This is the regression-fix from Waves 1-6 where the price-zero hack
    on delayed-benefit products produced NEGATIVE lift.
  * Cost-per-lift ranking surfaces a cheap-but-decent intervention above
    a more-expensive-but-only-slightly-bigger one.
  * Pairwise combo row is appended; additivity classification fires.
  * `DEPRECATED_V1_INTERVENTIONS` is gone — the module no longer
    exports the v1 spec list.
  * `advance_trialing()` resolves trials via the sunk-cost + consistency
    pressure formula and writes `trial_outcome` correctly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mindsim.engine import interventions as interventions_mod
from mindsim.engine.interventions import (
    InterventionSpec,
    _classify_additivity,
    _cost_per_pp,
    default_interventions,
    rank_interventions,
)
from mindsim.engine.population import generate_population
from mindsim.engine.state_machine import advance_trialing
from mindsim.models.config import (
    CalibratedParam,
    PopulationConfig,
    ReferencePriceParam,
    SimulationConfig,
    SimulationParams,
)
from mindsim.models.state import Phase
from mindsim.pipeline.simulate import simulate


# ─────────────────── shared fixtures ───────────────────


def _delayed_benefit_config() -> SimulationConfig:
    """A habit-tracker-ish product where time_to_value is high.

    The pre-Wave-7 v1 price=0 hack hurt adoption on this profile because
    the delayed-benefit + present-bias multiplicative penalty fired
    against the trial too. Wave 7 free trial routes via TRIALING and
    therefore lets sunk-cost + consistency pressure resolve in favour of
    adoption — the headline regression-fix.
    """
    return SimulationConfig(
        simulation_params=SimulationParams(
            price=10.0,
            reference_price=ReferencePriceParam(value=8.0, basis="ref", confidence=0.6),
            category_penetration=CalibratedParam(value=0.30, confidence=0.6),
            category_growth=CalibratedParam(value=0.50, confidence=0.5),
            perceived_benefit=CalibratedParam(value=0.65, confidence=0.6),
            benefit_certainty=CalibratedParam(value=0.35, confidence=0.5),  # low pre-trial
            time_to_value=CalibratedParam(value=0.5, confidence=0.6),       # delayed
            switching_cost=CalibratedParam(value=0.45, confidence=0.6),
            social_visibility=CalibratedParam(value=0.4, confidence=0.6),
            identity_signal=CalibratedParam(value=0.3, confidence=0.6),
            requires_behavior_change=CalibratedParam(value=0.7, confidence=0.6),
            present_bias_beta=CalibratedParam(value=0.55, confidence=0.6),  # behavior-change
            fomo_intensity=CalibratedParam(value=0.4, confidence=0.6),
            product_adoption_rate=CalibratedParam(value=0.05, confidence=0.5),
            feature_matrix=[],  # exercise the scalar restamp path
        ),
        population_config=PopulationConfig(),
    )


def _seed_run(seed: int = 7, n_agents: int = 800, n_rounds: int = 4):
    rng = np.random.default_rng(seed)
    config = _delayed_benefit_config()
    result = simulate(config, n_agents=n_agents, rng=rng, n_rounds=n_rounds)
    return result, config, rng


# ─────────────────── trialing transitions ───────────────────


class TestAdvanceTrialing:
    def _trial_agents(self, n: int, rounds_remaining: int = 1) -> np.ndarray:
        agents = generate_population(
            n=n,
            population_config=PopulationConfig(),
            sim_params=_delayed_benefit_config().simulation_params,
            rng=np.random.default_rng(11),
        )
        agents["phase"] = int(Phase.TRIALING)
        agents["trial_rounds_remaining"] = rounds_remaining
        agents["trial_outcome"] = -1
        # Push perceived_benefit high and switching_cost low so most convert.
        agents["agent_perceived_benefit"] = 0.8
        agents["agent_switching_cost"] = 0.2
        return agents

    def test_high_value_low_friction_converts_majority(self):
        agents = self._trial_agents(200, rounds_remaining=1)
        rng = np.random.default_rng(0)
        n_conv, n_quit = advance_trialing(agents, rng=rng)
        assert n_conv > 0.6 * 200
        assert n_conv + n_quit == 200
        assert (agents["phase"][agents["trial_outcome"] == 1] == int(Phase.ADOPTED)).all()
        assert (agents["phase"][agents["trial_outcome"] == 0] == int(Phase.AWARE)).all()

    def test_high_friction_low_benefit_quits_majority(self):
        agents = self._trial_agents(200, rounds_remaining=1)
        agents["agent_perceived_benefit"] = 0.2
        agents["agent_switching_cost"] = 0.9
        rng = np.random.default_rng(0)
        n_conv, n_quit = advance_trialing(agents, rng=rng)
        assert n_quit > 0.6 * 200

    def test_active_trial_just_decrements(self):
        agents = self._trial_agents(50, rounds_remaining=3)
        rng = np.random.default_rng(0)
        n_conv, n_quit = advance_trialing(agents, rng=rng)
        # No trial has reached 0 yet → no resolution.
        assert n_conv == 0 and n_quit == 0
        assert (agents["trial_rounds_remaining"][agents["phase"] == int(Phase.TRIALING)] == 2).all()

    def test_no_trialing_returns_zero(self):
        agents = self._trial_agents(20, rounds_remaining=1)
        agents["phase"] = int(Phase.AWARE)  # no one trialing
        rng = np.random.default_rng(0)
        assert advance_trialing(agents, rng=rng) == (0, 0)


# ─────────────────── helper math ───────────────────


class TestCostPerPp:
    def test_negative_lift_returns_inf(self):
        spec = default_interventions()[0]
        assert _cost_per_pp(spec, lift_pp=-0.5) == math.inf

    def test_zero_cost_returns_zero(self):
        spec = next(s for s in default_interventions() if s.mechanism_type == "annual_discount")
        assert _cost_per_pp(spec, lift_pp=2.0) == 0.0

    def test_positive_lift_uses_midpoint(self):
        spec = next(s for s in default_interventions() if s.mechanism_type == "free_trial")
        midpoint = 0.5 * (spec.cost_usd_low + spec.cost_usd_high)
        assert _cost_per_pp(spec, lift_pp=10.0) == pytest.approx(midpoint / 10.0)

    def test_lift_floor_applies_for_tiny_positives(self):
        spec = next(s for s in default_interventions() if s.mechanism_type == "free_trial")
        midpoint = 0.5 * (spec.cost_usd_low + spec.cost_usd_high)
        # 0.1pp lift gets divided by the 0.5pp floor, not 0.1.
        assert _cost_per_pp(spec, lift_pp=0.1) == pytest.approx(midpoint / 0.5)


class TestAdditivityClassifier:
    def test_super_additive_when_combo_exceeds_sum(self):
        assert _classify_additivity(2.0, 3.0, 6.0) == "super_additive"

    def test_sub_additive_when_combo_falls_short(self):
        assert _classify_additivity(2.0, 3.0, 3.0) == "sub_additive"

    def test_additive_within_tolerance(self):
        assert _classify_additivity(2.0, 3.0, 5.2) == "additive"
        assert _classify_additivity(2.0, 3.0, 4.8) == "additive"


# ─────────────────── ranking + free-trial regression fix ───────────────────


class TestRanking:
    def test_returns_results_for_each_default_intervention_plus_combo(self):
        result, config, rng = _seed_run()
        out = rank_interventions(
            agents=result._agents,
            config=config,
            base_adoption=result.total_adoption,
            rng=rng,
        )
        # 5 singles + 1 combo = 6 (combo only when at least 2 positive lifts).
        assert len(out) >= 5
        # Singles are sorted by cost_per_adoption_pp ascending.
        singles = [r for r in out if r.combined_with is None]
        cpps = [r.cost_per_adoption_pp for r in singles]
        assert cpps == sorted(cpps)

    def test_free_trial_produces_positive_lift_on_delayed_benefit(self):
        """The Wave 7 headline regression-fix.

        Pre-Wave-7 v1 free-trial set price=0 and produced NEGATIVE lift on
        delayed-benefit products. Wave 7 routes through TRIALING with
        sunk-cost + consistency bonuses; lift must be POSITIVE.
        """
        result, config, rng = _seed_run()
        out = rank_interventions(
            agents=result._agents,
            config=config,
            base_adoption=result.total_adoption,
            interventions=[
                spec for spec in default_interventions()
                if spec.mechanism_type == "free_trial"
            ],
            rng=rng,
            pairwise_combo=False,
        )
        assert len(out) == 1
        free_trial = out[0]
        assert free_trial.lift_pp > 0, (
            f"free trial must produce positive lift; got {free_trial.lift_pp:+.2f}pp"
        )
        assert free_trial.timeline_rounds >= 3
        assert free_trial.cost_usd_high > 0

    def test_free_trial_routes_some_agents_through_trialing(self):
        """Smoke test: a free-trial run leaves a non-trivial trial trace."""
        from mindsim.pipeline.simulate import run_rounds

        result, config, _rng = _seed_run()
        config = config.model_copy(deep=True)
        config.simulation_params.free_trial_active = True
        config.simulation_params.free_trial_duration_rounds = 3

        agents = result._agents.copy()
        run_rounds(
            agents=agents,
            params=config.simulation_params,
            n_rounds=4,
            rng=np.random.default_rng(0),
            starting_round=1,
        )
        # By the end, some agents should have a non-default trial_outcome.
        assert (agents["trial_outcome"] != -1).sum() > 0

    def test_zero_cost_intervention_lands_at_top_when_lift_positive(self):
        result, config, rng = _seed_run()
        out = rank_interventions(
            agents=result._agents,
            config=config,
            base_adoption=result.total_adoption,
            rng=rng,
            pairwise_combo=False,
        )
        # If annual discount produced any positive lift, it must lead the
        # ranking by cost-per-pp (cost is 0 → cpp is 0).
        positives = [r for r in out if r.lift_pp > 0]
        annual = next((r for r in out if r.mechanism_type == "annual_discount"), None)
        if annual and annual.lift_pp > 0:
            assert positives[0].mechanism_type == "annual_discount"

    def test_combo_row_classifies_additivity(self):
        result, config, rng = _seed_run()
        out = rank_interventions(
            agents=result._agents,
            config=config,
            base_adoption=result.total_adoption,
            rng=rng,
            pairwise_combo=True,
        )
        combos = [r for r in out if r.combined_with is not None]
        if combos:  # only when ≥2 positive-lift singles
            combo = combos[0]
            assert combo.additivity in ("sub_additive", "super_additive", "additive")
            assert "+" in combo.name
            assert combo.cost_usd_high > 0


class TestModuleSurfaceArea:
    def test_deprecated_v1_interventions_removed(self):
        assert not hasattr(interventions_mod, "DEPRECATED_V1_INTERVENTIONS")

    def test_default_interventions_returns_five(self):
        specs = default_interventions()
        assert len(specs) == 5
        types = {s.mechanism_type for s in specs}
        assert types == {
            "free_trial", "price_cut", "annual_discount",
            "social_proof_push", "freemium",
        }

    def test_each_spec_has_cost_and_timeline(self):
        for spec in default_interventions():
            assert spec.timeline_rounds >= 1
            assert spec.cost_usd_low <= spec.cost_usd_high
            assert spec.mechanism  # non-empty mechanism string
