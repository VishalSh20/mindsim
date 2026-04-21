"""Cluster assignment + cluster-local adoption rates (Wave 4).

Replaces the Wave 3 global `product_adoption_rate` scalar used as a
word-of-mouth proxy with cluster-local rates so social proof is computed
against each agent's actual network.

Fallback scheme until Wave 5 scrapers land: 6 deterministic clusters =
archetype-tier (early / mid / late) x income-tier (below / above median).

Cluster IDs are stable (0-5) so downstream consumers (RoundSnapshot,
forces.py) can index a length-6 rate / visibility array directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mindsim.models.state import ADOPTED_PHASES


N_CLUSTERS = 6

# Cluster id layout: archetype_tier (0..2) * 2 + income_tier (0..1).
#   0: early x below_median
#   1: early x above_median
#   2: mid   x below_median
#   3: mid   x above_median
#   4: late  x below_median
#   5: late  x above_median
CLUSTER_NAMES: tuple[str, ...] = (
    "early_low_income",
    "early_high_income",
    "mid_low_income",
    "mid_high_income",
    "late_low_income",
    "late_high_income",
)

# Per-cluster visibility factor. Multiplies social_visibility in the
# cluster-local social-proof formula and the WOM term in awareness
# promotion. Hardcoded defaults until Wave 5 pulls real community data.
#
# Rationale: early-adopter networks (tech Twitter, r/programming, HN)
# are loud and high-reach; late-majority / laggard networks (family
# groups, offline circles) are quiet and low-reach.
CLUSTER_VISIBILITY: np.ndarray = np.array(
    [
        0.70,  # early_low_income
        0.80,  # early_high_income  (tech Twitter tier)
        0.50,  # mid_low_income
        0.55,  # mid_high_income
        0.30,  # late_low_income    (quiet networks)
        0.35,  # late_high_income
    ],
    dtype=np.float64,
)


# Archetype id -> archetype tier (0=early, 1=mid, 2=late). Order must
# match ARCHETYPE_NAMES in models/config.py: innovator, early_adopter,
# early_majority, late_majority, laggard.
_ARCHETYPE_TIER: np.ndarray = np.array(
    [0, 0, 1, 2, 2], dtype=np.int8
)


@dataclass
class Cluster:
    """Structured cluster record for MarketContext.clusters."""

    id: int
    name: str
    visibility_factor: float
    size: int


def assign_clusters(
    archetype_ids: np.ndarray,
    incomes: np.ndarray,
) -> np.ndarray:
    """Deterministic cluster assignment from archetype + income.

    Args:
        archetype_ids: int array, values in [0, 5) (the 5 Rogers archetypes).
        incomes: float array, same length.

    Returns:
        uint8 array of cluster ids in [0, 6).
    """
    archetype_ids = np.asarray(archetype_ids, dtype=np.int8)
    incomes = np.asarray(incomes, dtype=np.float64)
    assert archetype_ids.shape == incomes.shape, "archetype_ids and incomes must have same length"

    arch_tier = _ARCHETYPE_TIER[archetype_ids]

    # Income tier is relative to THIS population's median so clusters stay
    # balanced across different income distributions.
    if incomes.size == 0:
        return np.zeros(0, dtype=np.uint8)
    median = float(np.median(incomes))
    income_tier = (incomes >= median).astype(np.int8)

    cluster_ids = (arch_tier * 2 + income_tier).astype(np.uint8)
    return cluster_ids


def compute_cluster_adoption_rates(
    agents: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    """Fraction ADOPTED or LOCKED_IN per cluster.

    Args:
        agents: agent array with `phase` and `cluster_id` fields.
        n_clusters: length of returned array. Defaults to 6.

    Returns:
        Length-`n_clusters` float64 array. Index = cluster_id. Empty
        clusters return 0.0 rather than NaN so callers can multiply
        without masking.
    """
    phase = agents["phase"]
    cluster_id = agents["cluster_id"].astype(np.int64)
    adopted_vals = np.array(ADOPTED_PHASES, dtype=phase.dtype)
    adopted = np.isin(phase, adopted_vals)

    rates = np.zeros(n_clusters, dtype=np.float64)
    # Bincount path: fast, vectorised.
    total_per_cluster = np.bincount(
        cluster_id, minlength=n_clusters
    ).astype(np.float64)
    adopted_per_cluster = np.bincount(
        cluster_id, weights=adopted.astype(np.float64), minlength=n_clusters
    )
    nonempty = total_per_cluster > 0
    rates[nonempty] = adopted_per_cluster[nonempty] / total_per_cluster[nonempty]
    return rates


def cluster_visibility_array() -> np.ndarray:
    """Return the static per-cluster visibility multiplier array."""
    return CLUSTER_VISIBILITY.copy()


def build_cluster_list(agents: np.ndarray) -> list[Cluster]:
    """Build a list of `Cluster` records for MarketContext.clusters.

    Sizes are computed from the current agent array; visibility is the
    static default from CLUSTER_VISIBILITY.
    """
    cluster_id = agents["cluster_id"].astype(np.int64)
    sizes = np.bincount(cluster_id, minlength=N_CLUSTERS)
    out: list[Cluster] = []
    for cid in range(N_CLUSTERS):
        out.append(
            Cluster(
                id=cid,
                name=CLUSTER_NAMES[cid],
                visibility_factor=float(CLUSTER_VISIBILITY[cid]),
                size=int(sizes[cid]),
            )
        )
    return out
