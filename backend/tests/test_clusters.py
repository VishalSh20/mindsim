"""Tests for Wave 4 cluster-local social proof + WOM.

Covers:
  - Cluster assignment determinism.
  - Cluster adoption-rate aggregation.
  - Two-cluster topology: adoption concentrated in cluster A produces
    stronger social-proof force for cluster-A agents than cluster-B agents.
  - promote_unaware_to_aware WOM is cluster-local when cluster_rates passed.
  - Fallback (cluster_rates=None) preserves the Wave 3 global formula.
"""
from __future__ import annotations

import numpy as np
import pytest

from mindsim.engine.clusters import (
    CLUSTER_VISIBILITY,
    N_CLUSTERS,
    assign_clusters,
    build_cluster_list,
    cluster_visibility_array,
    compute_cluster_adoption_rates,
)
from mindsim.engine.forces import compute_forces
from mindsim.engine.population import generate_population
from mindsim.engine.state_machine import promote_unaware_to_aware
from mindsim.models.config import SimulationConfig, SimulationParams
from mindsim.models.state import Phase


# ───────────────────────── Cluster assignment ─────────────────────────


class TestAssignClusters:
    def test_assignment_is_deterministic(self):
        archetype_ids = np.array([0, 1, 2, 3, 4, 2, 0], dtype=np.int8)
        incomes = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.float64)
        a = assign_clusters(archetype_ids, incomes)
        b = assign_clusters(archetype_ids, incomes)
        np.testing.assert_array_equal(a, b)

    def test_ids_in_range(self):
        rng = np.random.default_rng(0)
        archetype_ids = rng.integers(0, 5, size=500).astype(np.int8)
        incomes = rng.lognormal(10.5, 0.6, size=500)
        cids = assign_clusters(archetype_ids, incomes)
        assert cids.min() >= 0
        assert cids.max() < N_CLUSTERS

    def test_archetype_tier_monotonic(self):
        """Innovator (arch=0) always maps to an 'early' cluster (0 or 1).
        Laggard (arch=4) always maps to a 'late' cluster (4 or 5)."""
        archetype_ids = np.array([0, 4], dtype=np.int8)
        incomes = np.array([10_000, 100_000], dtype=np.float64)
        cids = assign_clusters(archetype_ids, incomes)
        assert cids[0] in (0, 1)
        assert cids[1] in (4, 5)

    def test_income_tier_uses_median(self):
        """Agents above population median get the '+1' (high-income) id."""
        archetype_ids = np.zeros(6, dtype=np.int8)  # all innovators
        incomes = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        cids = assign_clusters(archetype_ids, incomes)
        # median = 35 → first 3 below (id 0), last 3 above (id 1).
        np.testing.assert_array_equal(cids[:3], [0, 0, 0])
        np.testing.assert_array_equal(cids[3:], [1, 1, 1])

    def test_visibility_array_length(self):
        vis = cluster_visibility_array()
        assert vis.shape == (N_CLUSTERS,)
        assert (vis > 0).all()


# ─────────────────── Cluster adoption rate aggregation ───────────────────


class TestClusterAdoptionRates:
    def _make_agents(self, phases, cluster_ids):
        """Build minimal agent array with phase + cluster_id only."""
        from mindsim.engine.population import AGENT_DTYPE
        n = len(phases)
        agents = np.zeros(n, dtype=AGENT_DTYPE)
        agents["phase"] = np.asarray(phases, dtype=np.int8)
        agents["cluster_id"] = np.asarray(cluster_ids, dtype=np.uint8)
        return agents

    def test_all_adopted_one_cluster(self):
        agents = self._make_agents(
            phases=[int(Phase.ADOPTED)] * 5 + [int(Phase.UNAWARE)] * 5,
            cluster_ids=[0] * 5 + [1] * 5,
        )
        rates = compute_cluster_adoption_rates(agents)
        assert rates[0] == pytest.approx(1.0)
        assert rates[1] == pytest.approx(0.0)
        assert (rates[2:] == 0.0).all()

    def test_locked_in_counts_as_adopted(self):
        agents = self._make_agents(
            phases=[int(Phase.LOCKED_IN), int(Phase.ADOPTED), int(Phase.CHURNED)],
            cluster_ids=[0, 0, 0],
        )
        rates = compute_cluster_adoption_rates(agents)
        # 2 adopter-equivalents (LOCKED_IN + ADOPTED) out of 3.
        assert rates[0] == pytest.approx(2.0 / 3.0)

    def test_empty_cluster_rate_is_zero(self):
        agents = self._make_agents(
            phases=[int(Phase.ADOPTED), int(Phase.ADOPTED)],
            cluster_ids=[3, 3],
        )
        rates = compute_cluster_adoption_rates(agents)
        assert rates[0] == 0.0
        assert rates[3] == pytest.approx(1.0)


# ────────────── Build cluster list (for MarketContext.clusters) ──────────────


class TestBuildClusterList:
    def test_sizes_sum_to_population(self):
        rng = np.random.default_rng(7)
        agents = generate_population(n=500, rng=rng)
        clusters = build_cluster_list(agents)
        assert len(clusters) == N_CLUSTERS
        assert sum(c.size for c in clusters) == 500

    def test_visibility_matches_static(self):
        rng = np.random.default_rng(7)
        agents = generate_population(n=200, rng=rng)
        clusters = build_cluster_list(agents)
        for i, c in enumerate(clusters):
            assert c.visibility_factor == pytest.approx(CLUSTER_VISIBILITY[i])


# ─────────────────── Two-cluster topology integration ───────────────────


class TestTwoClusterTopology:
    """Critical-mass integration: adoption inside cluster A should produce
    stronger social-proof force for A-agents than for B-agents."""

    def test_social_proof_higher_in_adopted_cluster(self):
        rng = np.random.default_rng(123)
        agents = generate_population(n=2000, rng=rng)

        # Force two test clusters: 0 (early_low) and 1 (early_high). All
        # aware so compute_forces produces values for everyone.
        agents["phase"] = int(Phase.CONSIDERING)
        agents["aware"] = True
        agents["awareness_strength"] = 1.0

        # Pin agents to cluster 0 vs cluster 1 via cluster_id override.
        # Even split.
        agents["cluster_id"] = np.where(
            np.arange(len(agents)) % 2 == 0, 0, 1
        ).astype(np.uint8)

        # cluster_rates: cluster 0 fully adopted (1.0), cluster 1 cold (0.0).
        cluster_rates = np.zeros(N_CLUSTERS, dtype=np.float64)
        cluster_rates[0] = 1.0

        params = SimulationParams()
        forces = compute_forces(agents, params, cluster_rates=cluster_rates)

        sp = forces["social_proof"]
        cid = agents["cluster_id"]
        sp_a = sp[cid == 0]
        sp_b = sp[cid == 1]

        assert np.isfinite(sp_a).all()
        assert np.isfinite(sp_b).all()
        # Cluster A agents face real social proof; B agents near zero.
        assert sp_a.mean() > sp_b.mean() + 0.05, (
            f"Expected cluster-A social proof >> B, got "
            f"A={sp_a.mean():.3f} B={sp_b.mean():.3f}"
        )
        # And B should be near zero (no adopters in their cluster).
        assert sp_b.mean() < 1e-6


# ──────────────── promote_unaware_to_aware WOM cluster-local ────────────────


class TestClusterLocalWOM:
    def test_hot_cluster_promotes_faster_than_cold(self):
        """Given two disjoint clusters, the one with high adoption should
        convert more UNAWARE agents to AWARE per round than the cold one.

        Uses marketing_reach=0 so all promotion comes from WOM.
        """
        rng = np.random.default_rng(42)
        agents = generate_population(n=2000, rng=rng)
        agents["phase"] = int(Phase.UNAWARE)
        agents["awareness_strength"] = 0.0
        # Half cluster 0, half cluster 1.
        agents["cluster_id"] = np.where(
            np.arange(len(agents)) % 2 == 0, 0, 1
        ).astype(np.uint8)

        cluster_rates = np.zeros(N_CLUSTERS, dtype=np.float64)
        cluster_rates[0] = 0.8  # hot

        promote_unaware_to_aware(
            agents,
            product_adoption_rate=0.0,  # unused in cluster path
            rng=rng,
            marketing_reach=0.0,
            cluster_rates=cluster_rates,
        )

        hot_aware = int(
            (
                (agents["cluster_id"] == 0)
                & (agents["phase"] == int(Phase.AWARE))
            ).sum()
        )
        cold_aware = int(
            (
                (agents["cluster_id"] == 1)
                & (agents["phase"] == int(Phase.AWARE))
            ).sum()
        )
        assert hot_aware > cold_aware * 3, (
            f"Expected hot cluster to dominate WOM promotion; "
            f"hot={hot_aware} cold={cold_aware}"
        )
        # Cold cluster should see ~zero (no WOM, no marketing).
        assert cold_aware == 0

    def test_fallback_scalar_path_unchanged(self):
        """cluster_rates=None falls back to the global scalar formula.
        Non-zero product_adoption_rate + zero marketing still promotes
        a visible fraction.
        """
        rng = np.random.default_rng(1)
        agents = generate_population(n=1000, rng=rng)
        agents["phase"] = int(Phase.UNAWARE)
        agents["awareness_strength"] = 0.0

        promoted = promote_unaware_to_aware(
            agents,
            product_adoption_rate=0.5,
            rng=rng,
            marketing_reach=0.0,
            cluster_rates=None,
        )
        assert promoted > 0

    def test_forces_fallback_scalar_unchanged(self):
        """compute_forces without cluster_rates must preserve the old
        behaviour: social_proof > 0 when product_adoption_rate > 0."""
        rng = np.random.default_rng(9)
        agents = generate_population(n=500, rng=rng)
        agents["phase"] = int(Phase.CONSIDERING)
        agents["aware"] = True
        agents["awareness_strength"] = 1.0

        params = SimulationParams()
        # Bump global scalar so social proof is clearly non-zero.
        params.product_adoption_rate.value = 0.3

        forces = compute_forces(agents, params, cluster_rates=None)
        sp = forces["social_proof"]
        assert np.isfinite(sp).all()
        assert sp.mean() > 0.0
