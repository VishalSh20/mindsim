"""Per-agent tiered competitor knowledge (Wave 5).

Each agent sees the market through a personalised lens:
  - primary competitor: drawn by market-share weight (the dominant tool
    the agent actually considers switching from / against).
  - secondary competitors: drawn by cluster visibility × archetype
    breadth (tools they've heard of but don't live in).
  - unaware competitors: the rest.

Per-agent reference price:
    0.6 × primary.price + 0.3 × mean(secondary.price) + 0.1 × category_default

Design note: we do NOT extend AGENT_DTYPE with competitor-id fields.
Instead we stamp the derived `agent_reference_price` directly. Keeps the
dtype stable (critical for Wave 6 session-state serialisation) and
spares us a dtype migration.

Fallback: when no competitors carry `market_share` (e.g., Wave 0-4
research that didn't run A2), the caller leaves the archetype-based
reference-price stamping in `population.py` untouched. This module is
a strict upgrade path, not a replacement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from mindsim.engine.clusters import CLUSTER_VISIBILITY
from mindsim.models.market import VerifiedCompetitor

logger = logging.getLogger(__name__)


# How many secondary competitors each agent "knows about" — cap so
# laggards aren't handed the whole catalogue.
MAX_SECONDARY = 3

# Weights in per-agent reference price formula.
W_PRIMARY = 0.6
W_SECONDARY = 0.3
W_CATEGORY_DEFAULT = 0.1


@dataclass
class TierAssignment:
    """Per-agent competitor tier IDs (indexes into the `competitors` list).

    `primary_idx`: -1 if no primary could be drawn.
    `secondary_idx`: list of indices, up to MAX_SECONDARY. Sorted.
    `unaware_idx`: remainder of competitor indices not in primary/secondary.
    `reference_price`: float — the agent's derived reference price.
    """

    primary_idx: int
    secondary_idx: list[int]
    unaware_idx: list[int]
    reference_price: float


def _normalise_shares(competitors: list[VerifiedCompetitor]) -> np.ndarray:
    """Return a length-N probability vector from market_share values.

    Competitors with `market_share=None` get uniform weight split of the
    unallocated remainder. If no shares at all are given, returns a
    uniform distribution. Always sums to 1.0.
    """
    n = len(competitors)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    shares = np.zeros(n, dtype=np.float64)
    have_share = np.zeros(n, dtype=bool)
    for i, c in enumerate(competitors):
        if c.market_share is not None and 0.0 <= c.market_share <= 1.0:
            shares[i] = float(c.market_share)
            have_share[i] = True

    explicit_total = shares.sum()
    if explicit_total >= 1.0:
        # Normalise explicit-only.
        return shares / explicit_total

    # Distribute the remainder evenly across competitors without a share.
    remainder = 1.0 - explicit_total
    missing = (~have_share).sum()
    if missing == 0:
        # All competitors had shares but sum <1 — normalise upward.
        return shares / explicit_total if explicit_total > 0 else np.full(n, 1.0 / n)
    shares[~have_share] = remainder / missing
    return shares


def assign_competitor_tiers(
    agents: np.ndarray,
    competitors: list[VerifiedCompetitor],
    category_default_price: float,
    rng: np.random.Generator | None = None,
) -> list[TierAssignment]:
    """Assign primary / secondary / unaware competitors per agent.

    Deterministic per (agent archetype, cluster, rng seed). Returns a
    list of TierAssignment, one per agent, ordered as `agents`.
    """
    n_agents = len(agents)
    n_comp = len(competitors)
    if rng is None:
        rng = np.random.default_rng(0)

    if n_comp == 0:
        # Nothing to assign — every agent anchors to category default only.
        return [
            TierAssignment(
                primary_idx=-1,
                secondary_idx=[],
                unaware_idx=[],
                reference_price=category_default_price,
            )
            for _ in range(n_agents)
        ]

    share_probs = _normalise_shares(competitors)

    # Secondary-tier probability = normalised "visibility × breadth" score.
    # Breadth: how broadly named a competitor is across archetypes. Use
    # `market_share` as a proxy when available; else uniform. Cluster
    # visibility then multiplies by the AGENT's cluster — so laggard-
    # cluster agents see fewer secondary competitors.
    breadth = np.where(share_probs > 0, share_probs, 1.0 / n_comp)

    # Primary prices, for the formula.
    prices = np.array(
        [c.confirmed_price if c.confirmed_price is not None else category_default_price
         for c in competitors],
        dtype=np.float64,
    )

    cluster_ids = agents["cluster_id"].astype(np.int64)

    out: list[TierAssignment] = []
    for i in range(n_agents):
        cid = int(cluster_ids[i])
        vis = float(CLUSTER_VISIBILITY[cid]) if 0 <= cid < len(CLUSTER_VISIBILITY) else 0.5

        # Primary: sample by market-share weight.
        primary = int(rng.choice(n_comp, p=share_probs))

        # Secondary: sample (up to MAX_SECONDARY) from the remainder with
        # prob ∝ breadth[c] × vis. Cold-cluster agents end up with fewer.
        remaining_mask = np.ones(n_comp, dtype=bool)
        remaining_mask[primary] = False
        if remaining_mask.any():
            secondary_weights = breadth * vis * remaining_mask.astype(np.float64)
            secondary_weights = np.maximum(secondary_weights, 0.0)
            # Each remaining competitor independently qualifies as
            # secondary with probability proportional to its score. We
            # cap at MAX_SECONDARY by taking the top-k candidates that
            # cleared a per-candidate random draw.
            draws = rng.random(n_comp)
            eligible = (draws < secondary_weights) & remaining_mask
            if eligible.sum() > MAX_SECONDARY:
                # Top MAX_SECONDARY by weight among eligible.
                scored = np.where(eligible, secondary_weights, -1.0)
                top = np.argsort(scored)[::-1][:MAX_SECONDARY]
                secondary = sorted(int(k) for k in top)
            else:
                secondary = sorted(int(k) for k in np.flatnonzero(eligible))
        else:
            secondary = []

        assigned = {primary, *secondary}
        unaware = sorted(set(range(n_comp)) - assigned)

        ref_price = derive_agent_reference_price(
            primary_price=float(prices[primary]),
            secondary_prices=[float(prices[k]) for k in secondary],
            category_default=category_default_price,
        )

        out.append(
            TierAssignment(
                primary_idx=primary,
                secondary_idx=secondary,
                unaware_idx=unaware,
                reference_price=ref_price,
            )
        )

    return out


def derive_agent_reference_price(
    primary_price: float,
    secondary_prices: list[float],
    category_default: float,
) -> float:
    """Weighted combination per tier (0.6 / 0.3 / 0.1)."""
    if secondary_prices:
        sec_mean = float(np.mean(secondary_prices))
    else:
        # No secondary tier → roll its weight into primary + default.
        sec_mean = (primary_price + category_default) / 2.0
    return (
        W_PRIMARY * primary_price
        + W_SECONDARY * sec_mean
        + W_CATEGORY_DEFAULT * category_default
    )


def apply_tiered_reference_prices(
    agents: np.ndarray,
    competitors: list[VerifiedCompetitor],
    category_default_price: float,
    rng: np.random.Generator | None = None,
) -> None:
    """Stamp `agent_reference_price` from tier assignment in-place.

    No-op when no competitor carries `market_share` (keeps the archetype-
    based stamping from `population.py` as the fallback).
    """
    if not competitors:
        return
    has_any_share = any(
        c.market_share is not None and c.market_share > 0 for c in competitors
    )
    if not has_any_share:
        logger.info(
            "No competitor market_share data; retaining archetype-based reference prices"
        )
        return

    tiers = assign_competitor_tiers(
        agents, competitors, category_default_price, rng=rng
    )
    for i, t in enumerate(tiers):
        agents["agent_reference_price"][i] = np.float32(t.reference_price)
