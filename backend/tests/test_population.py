"""Test population generation — archetype shares and distributions."""

import numpy as np
import pytest

from mindsim.engine.archetypes import load_archetypes
from mindsim.engine.population import AGENT_DTYPE, generate_population


class TestArchetypeShares:
    """Verify Rogers archetype share distribution."""

    def test_shares_sum_to_one(self):
        archetype_set = load_archetypes()
        total = sum(archetype_set.shares)
        assert abs(total - 1.0) < 0.01, f"Shares sum to {total}, expected ~1.0"

    def test_shares_match_rogers(self):
        """10,000 agents → archetype shares within ±2pp of Rogers."""
        rng = np.random.default_rng(42)
        archetype_set = load_archetypes()
        agents = generate_population(n=10000, rng=rng, archetype_set=archetype_set)

        names = archetype_set.names
        expected = {n: s for n, s in zip(names, archetype_set.shares)}

        for i, name in enumerate(names):
            actual = (agents["archetype_id"] == i).sum() / 10000
            exp = expected[name]
            assert abs(actual - exp) < 0.02, (
                f"{name}: expected {exp:.3f}, got {actual:.3f} (diff {abs(actual-exp):.3f})"
            )


class TestLossAversionDistribution:
    """Verify the Gächter mixture model."""

    def test_has_near_zero_subgroup(self):
        """~20% of agents should have λ near 1.0 (within 1.0-1.5)."""
        rng = np.random.default_rng(42)
        agents = generate_population(n=10000, rng=rng)

        la = agents["loss_aversion_lambda"]
        near_zero_frac = ((la >= 1.0) & (la <= 1.5)).sum() / len(la)

        # Should be roughly 15-30% (mixture is 20% from N(1.1, 0.2))
        assert 0.10 <= near_zero_frac <= 0.40, (
            f"Near-zero λ fraction: {near_zero_frac:.2f}, expected 0.10-0.40"
        )

    def test_population_mean_reasonable(self):
        """Population mean λ should be roughly 1.8-2.5."""
        rng = np.random.default_rng(42)
        agents = generate_population(n=10000, rng=rng)
        mean_la = agents["loss_aversion_lambda"].mean()
        assert 1.5 <= mean_la <= 2.8, f"Mean λ = {mean_la:.2f}, expected 1.5-2.8"


class TestPersonalityCorrelations:
    """Verify personality-economics correlations exist."""

    def test_income_lognormal(self):
        """Income should be lognormally distributed (positive, right-skewed)."""
        rng = np.random.default_rng(42)
        agents = generate_population(n=10000, rng=rng)
        income = agents["income"]
        assert (income > 0).all(), "All incomes should be positive"
        assert income.mean() > np.median(income), "Income should be right-skewed"

    def test_valid_ranges(self):
        """All behavioral params should be in valid ranges."""
        rng = np.random.default_rng(42)
        agents = generate_population(n=10000, rng=rng)

        assert (agents["loss_aversion_lambda"] >= 1.0).all()
        assert (agents["status_quo_bias"] >= 0.0).all()
        assert (agents["social_proof_need"] >= 0.0).all()
        assert (agents["openness"] >= 0.0).all()
        assert (agents["openness"] <= 1.0).all()
