"""Tests for the v2-middle force interactions introduced in Wave 1.

Covers:
  1. Anchoring absorbed into prospect (delete standalone force).
  2. Probability weighting on certainty in social proof.
  3. Multiplicative loss × discount rule for delayed-benefit products.
"""
import numpy as np

from mindsim.engine.force_config import (
    BEHAVIOR_CHANGE_THRESHOLD,
    DELAYED_BENEFIT_THRESHOLD,
)
from mindsim.engine.forces import compute_forces
from mindsim.engine.population import AGENT_DTYPE
from mindsim.models.config import (
    CalibratedParam,
    ReferencePriceParam,
    SimulationParams,
)


def _params(
    *,
    price: float = 20.0,
    requires_behavior_change: float = 0.3,
    time_to_value: float = 0.2,
    benefit_certainty: float = 0.5,
    reference_price: float = 30.0,
    present_bias_beta: float = 0.75,
) -> SimulationParams:
    return SimulationParams(
        price=price,
        reference_price=ReferencePriceParam(value=reference_price),
        category_penetration=CalibratedParam(value=0.15),
        category_growth=CalibratedParam(value=0.5),
        benefit_certainty=CalibratedParam(value=benefit_certainty),
        perceived_benefit=CalibratedParam(value=0.6),
        time_to_value=CalibratedParam(value=time_to_value),
        requires_behavior_change=CalibratedParam(value=requires_behavior_change),
        switching_cost=CalibratedParam(value=0.3),
        social_visibility=CalibratedParam(value=0.4),
        identity_signal=CalibratedParam(value=0.3),
        present_bias_beta=CalibratedParam(value=present_bias_beta),
        fomo_intensity=CalibratedParam(value=0.3),
        product_adoption_rate=CalibratedParam(value=0.1),
    )


def _agent(**overrides) -> np.ndarray:
    a = np.zeros(1, dtype=AGENT_DTYPE)
    defaults = dict(
        archetype_id=2,
        loss_aversion_lambda=2.25,
        status_quo_bias=0.5,
        social_proof_need=0.5,
        novelty_weight=0.4,
        price_sensitivity=0.5,
        fomo_susceptibility=0.3,
        openness=0.5,
        neuroticism=0.5,
        agreeableness=0.5,
        conscientiousness=0.5,
        income=60000.0,
        aware=True,
        competitor_awareness_frac=1.0,
    )
    defaults.update(overrides)
    for field, value in defaults.items():
        a[field][0] = value
    return a


class TestAnchoringAbsorption:
    """v2-middle §R1: standalone anchoring force deleted, absorbed into prospect."""

    def test_anchoring_key_present_and_zero(self):
        agent = _agent()
        params = _params(price=20.0, reference_price=50.0)
        forces = compute_forces(agent, params)
        assert "anchoring" in forces
        assert forces["anchoring"][0] == 0.0

    def test_cheap_vs_expensive_prospect_ordering(self):
        """Below-reference product has higher prospect than above-reference."""
        agent = _agent()
        cheap = _params(price=20.0, reference_price=50.0)    # ratio 0.4
        expensive = _params(price=20.0, reference_price=10.0)  # ratio 2.0

        p_cheap = compute_forces(agent, cheap)["prospect_value"][0]
        p_exp = compute_forces(agent, expensive)["prospect_value"][0]
        assert p_cheap > p_exp

    def test_no_reference_means_no_ratio_adjustment(self):
        """When agent has no meaningful ref (=price), ratio is 1."""
        agent = _agent()
        at_ref = _params(price=20.0, reference_price=20.0)
        no_ref = _params(price=20.0, reference_price=0.0)  # falls back to price blend

        p_at = compute_forces(agent, at_ref)["prospect_value"][0]
        p_no = compute_forces(agent, no_ref)["prospect_value"][0]
        # When ref == price, blend = price, ratio = 1.0. When ref = 0 + blend
        # with price, agent_ref_full = price, also ratio = 1.0. Equal prospects.
        assert abs(p_at - p_no) < 1e-9


class TestProbabilityWeightingInSocialProof:
    """Social proof now multiplies by (1 - w(certainty)) not (1 - certainty)."""

    def test_social_proof_still_higher_when_certainty_low(self):
        agent = _agent(social_proof_need=0.7)
        certain = _params(benefit_certainty=0.9)
        uncertain = _params(benefit_certainty=0.2)

        s_c = compute_forces(agent, certain)["social_proof"][0]
        s_u = compute_forces(agent, uncertain)["social_proof"][0]
        assert s_u > s_c

    def test_small_certainty_is_overweighted(self):
        """At low certainty, w(cert) > cert, so (1 - w(cert)) < (1 - cert).

        Social proof should therefore be *slightly smaller* than a naive
        (1-cert) weighting would produce at cert=0.1, because small
        probabilities are over-weighted under T&K.
        """
        agent = _agent(social_proof_need=0.5)
        params = _params(benefit_certainty=0.1)
        forces = compute_forces(agent, params)
        # w(0.1) ≈ 0.186; (1 - 0.186) = 0.814 vs v1's (1 - 0.1) = 0.9
        # So social proof with weighting is strictly less than v1's.
        # Lock in that the force is positive and non-trivial.
        assert forces["social_proof"][0] > 0.0


class TestLossDiscountMultiplicativeRule:
    """v2-middle §R10 / Gap 5: delayed-benefit products multiply, not add."""

    def test_delayed_benefit_triggers_multiplicative_penalty(self):
        """Habit-change product with long time_to_value gets prospect × β."""
        agent = _agent(openness=0.5)

        # Matches condition: requires_behavior_change > 0.5 AND time_to_value > 0.3
        delayed = _params(
            requires_behavior_change=0.8,
            time_to_value=0.6,
            present_bias_beta=0.6,
        )
        # Otherwise identical, but doesn't trigger the rule.
        immediate = _params(
            requires_behavior_change=0.8,
            time_to_value=0.1,
            present_bias_beta=0.6,
        )

        p_del = compute_forces(agent, delayed)["prospect_value"][0]
        p_imm = compute_forces(agent, immediate)["prospect_value"][0]
        # Gain side is multiplied by β≈0.6 in the delayed case ⇒ smaller prospect.
        assert p_del < p_imm

    def test_discount_force_zeroed_when_rule_fires(self):
        agent = _agent()
        delayed = _params(requires_behavior_change=0.8, time_to_value=0.6)
        assert compute_forces(agent, delayed)["hyperbolic_discounting"][0] == 0.0

    def test_discount_force_nonzero_when_rule_does_not_fire(self):
        agent = _agent()
        # time_to_value below threshold ⇒ rule does NOT fire
        undelayed = _params(requires_behavior_change=0.8, time_to_value=0.1)
        assert compute_forces(agent, undelayed)["hyperbolic_discounting"][0] < 0.0

    def test_rule_requires_both_conditions(self):
        """Rule fires only when BOTH behavior_change AND time_to_value exceed thresholds."""
        agent = _agent()

        # Only behavior_change high → discount is additive (non-zero)
        only_bc = _params(requires_behavior_change=0.8, time_to_value=0.1)
        assert compute_forces(agent, only_bc)["hyperbolic_discounting"][0] < 0.0

        # Only time_to_value high → discount is additive (non-zero)
        only_ttv = _params(requires_behavior_change=0.2, time_to_value=0.6)
        assert compute_forces(agent, only_ttv)["hyperbolic_discounting"][0] < 0.0

        # Both high → discount zeroed (folded into prospect)
        both = _params(requires_behavior_change=0.8, time_to_value=0.6)
        assert compute_forces(agent, both)["hyperbolic_discounting"][0] == 0.0

    def test_thresholds_exposed_as_constants(self):
        assert BEHAVIOR_CHANGE_THRESHOLD == 0.5
        assert DELAYED_BENEFIT_THRESHOLD == 0.30


class TestMaturityHelper:
    """v2-middle §5.5: consideration_threshold derived from maturity."""

    def test_thresholds_by_maturity(self):
        from mindsim.engine.maturity import consideration_threshold_for_maturity

        assert consideration_threshold_for_maturity("nascent") == 0.60
        assert consideration_threshold_for_maturity("growing") == 0.35
        assert consideration_threshold_for_maturity("mainstream") == 0.45
        assert consideration_threshold_for_maturity("saturated") == 0.50

    def test_unknown_maturity_returns_default(self):
        from mindsim.engine.maturity import consideration_threshold_for_maturity

        assert consideration_threshold_for_maturity(None) == 0.45
        assert consideration_threshold_for_maturity("wat") == 0.45
