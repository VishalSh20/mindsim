"""Tiered competitor knowledge tests (Wave 5 Phase E).

Validates:
  - primary distribution ≈ market-share weights (±5pp on N=2000).
  - reference-price formula (0.6/0.3/0.1).
  - laggard-cluster agents see fewer secondary competitors on average.
  - fallback no-op when no market_share data present.
  - determinism with fixed RNG seed.
"""
from __future__ import annotations

import numpy as np
import pytest

from mindsim.engine.competitor_tiers import (
    MAX_SECONDARY,
    apply_tiered_reference_prices,
    assign_competitor_tiers,
    derive_agent_reference_price,
)
from mindsim.engine.population import generate_population
from mindsim.models.market import VerifiedCompetitor


def _comps():
    return [
        VerifiedCompetitor(name="Leader", confirmed_price=40.0, market_share=0.5),
        VerifiedCompetitor(name="Challenger", confirmed_price=20.0, market_share=0.3),
        VerifiedCompetitor(name="Niche1", confirmed_price=10.0, market_share=0.1),
        VerifiedCompetitor(name="Niche2", confirmed_price=5.0, market_share=0.1),
    ]


class TestReferencePriceFormula:
    def test_exact_weighted_combination(self):
        # primary=40, secondary=[20,10] mean=15, default=25
        # 0.6*40 + 0.3*15 + 0.1*25 = 24 + 4.5 + 2.5 = 31.0
        out = derive_agent_reference_price(
            primary_price=40.0,
            secondary_prices=[20.0, 10.0],
            category_default=25.0,
        )
        assert out == pytest.approx(31.0)

    def test_no_secondary_falls_back_to_primary_default_mean(self):
        # secondary empty → sec_mean = (primary+default)/2 = (40+20)/2 = 30
        # 0.6*40 + 0.3*30 + 0.1*20 = 24 + 9 + 2 = 35
        out = derive_agent_reference_price(40.0, [], 20.0)
        assert out == pytest.approx(35.0)


class TestPrimaryDistribution:
    def test_primary_matches_market_share_within_5pp(self):
        rng = np.random.default_rng(42)
        agents = generate_population(n=2000, rng=rng)
        comps = _comps()
        tiers = assign_competitor_tiers(
            agents, comps, category_default_price=25.0, rng=rng,
        )
        primary_ids = np.array([t.primary_idx for t in tiers])
        for i, c in enumerate(comps):
            observed = (primary_ids == i).mean()
            assert abs(observed - c.market_share) <= 0.05, (
                f"{c.name}: expected share {c.market_share}, got {observed:.3f}"
            )


class TestLaggardClustersFewerSecondary:
    def test_cold_cluster_sees_fewer_secondary_on_average(self):
        rng = np.random.default_rng(7)
        agents = generate_population(n=2000, rng=rng)
        comps = _comps()
        tiers = assign_competitor_tiers(
            agents, comps, category_default_price=25.0, rng=rng,
        )
        secondary_counts = np.array([len(t.secondary_idx) for t in tiers])
        cluster_ids = agents["cluster_id"].astype(np.int64)
        # Cluster ids 0,1 = early (high visibility); 4,5 = late (low visibility).
        early_mask = np.isin(cluster_ids, [0, 1])
        late_mask = np.isin(cluster_ids, [4, 5])
        # Both masks should be non-empty.
        assert early_mask.sum() > 100 and late_mask.sum() > 100
        early_avg = secondary_counts[early_mask].mean()
        late_avg = secondary_counts[late_mask].mean()
        assert early_avg > late_avg, (
            f"Early clusters should see more secondary comps: "
            f"early={early_avg:.2f} late={late_avg:.2f}"
        )


class TestDeterminism:
    def test_same_seed_same_assignment(self):
        rng_a = np.random.default_rng(123)
        rng_b = np.random.default_rng(123)
        agents_a = generate_population(n=500, rng=rng_a)
        agents_b = generate_population(n=500, rng=rng_b)
        # Re-seed for the tier call too.
        t_a = assign_competitor_tiers(
            agents_a, _comps(), 25.0, rng=np.random.default_rng(99)
        )
        t_b = assign_competitor_tiers(
            agents_b, _comps(), 25.0, rng=np.random.default_rng(99)
        )
        for a, b in zip(t_a, t_b):
            assert a.primary_idx == b.primary_idx
            assert a.secondary_idx == b.secondary_idx
            assert a.reference_price == pytest.approx(b.reference_price)


class TestApplyFallback:
    def test_no_market_share_is_noop(self):
        rng = np.random.default_rng(5)
        agents = generate_population(n=300, rng=rng)
        before = agents["agent_reference_price"].copy()

        comps_no_share = [
            VerifiedCompetitor(name="X", confirmed_price=10.0),
            VerifiedCompetitor(name="Y", confirmed_price=20.0),
        ]
        apply_tiered_reference_prices(
            agents, comps_no_share, category_default_price=15.0, rng=rng,
        )
        np.testing.assert_array_equal(agents["agent_reference_price"], before)

    def test_with_market_share_modifies_reference_price(self):
        rng = np.random.default_rng(5)
        agents = generate_population(n=300, rng=rng)
        before = agents["agent_reference_price"].copy()
        apply_tiered_reference_prices(
            agents, _comps(), category_default_price=25.0, rng=rng,
        )
        after = agents["agent_reference_price"]
        # At least some agents should now have a different reference price.
        assert (after != before).any()
        # And all values should be within the plausible span of competitor prices.
        assert after.min() >= 5.0 and after.max() <= 40.0


class TestNoCompetitors:
    def test_empty_competitor_list_anchors_to_default(self):
        rng = np.random.default_rng(0)
        agents = generate_population(n=10, rng=rng)
        tiers = assign_competitor_tiers(agents, [], category_default_price=42.0, rng=rng)
        assert all(t.primary_idx == -1 for t in tiers)
        assert all(t.reference_price == 42.0 for t in tiers)


class TestMaxSecondaryCap:
    def test_secondary_capped_at_max(self):
        rng = np.random.default_rng(11)
        agents = generate_population(n=500, rng=rng)
        # 8 competitors all with shares → many eligible secondary candidates.
        many_comps = [
            VerifiedCompetitor(name=f"C{i}", confirmed_price=10.0 + i, market_share=0.125)
            for i in range(8)
        ]
        tiers = assign_competitor_tiers(agents, many_comps, 25.0, rng=rng)
        max_secondary_seen = max(len(t.secondary_idx) for t in tiers)
        assert max_secondary_seen <= MAX_SECONDARY
