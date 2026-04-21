"""Per-round phase transitions for the v2-middle multi-round simulation.

MECHANICS-v2.md §Phase Transitions is the source. Every function here
operates vectorised on the full agent array. No Python loops over agents.

Transitions applied each round, in this order:

  1. decay_awareness        : awareness_strength *= (1 - decay); cross-below
                              threshold reverts to UNAWARE.
  2. promote_unaware_to_aware: marketing + word-of-mouth reach (UNAWARE → AWARE,
                              awareness_strength seeded to 0.5).
  3. promote_aware_to_considering: awareness_strength > consideration_threshold
                              (from maturity helper) AND relevance check.
  4. decide_considering     : compute forces + logistic → AWARE→ADOPTED
                              (direct — trialing phase is wired but unused
                              until Wave 7 adds the free-trial mechanism).
  5. advance_adopted         : investment_depth accumulates; crosses lock_in
                              threshold → LOCKED_IN. Basic churn check moves
                              some adopted agents to CHURNED.

The `aware` boolean is re-synced with `phase != UNAWARE` at the end of
each round so the existing force-computation slicing in forces.py still
works unchanged.
"""
from __future__ import annotations

import logging

import numpy as np

from mindsim.engine.clusters import cluster_visibility_array
from mindsim.engine.feature_forces import compute_feature_gain
from mindsim.engine.probability_weighting import probability_weight
from mindsim.models.config import SimulationParams
from mindsim.models.state import (
    ADOPTED_PHASES,
    AWARE_PHASES,
    DECISION_PHASES,
    Phase,
)

logger = logging.getLogger(__name__)


# ──────────────────────── Tunable constants ────────────────────────
# Kept here (not in force_config) since these are *transition* constants,
# not force-interaction thresholds.
AWARENESS_DECAY_PER_ROUND = 0.10       # 10% loss per round for agents who don't re-engage
AWARENESS_REVERT_THRESHOLD = 0.05      # below this → back to UNAWARE
MARKETING_REACH_PER_ROUND = 0.05       # base P(newly aware) per round
WOM_MULTIPLIER = 0.5                    # word-of-mouth weight on aware-adoption
INITIAL_AWARENESS_STRENGTH = 0.5        # seeded when UNAWARE → AWARE
CONSIDERATION_STRENGTH_GAIN = 0.20      # added to strength when AWARE is re-exposed
INVESTMENT_GROWTH_PER_ROUND = 0.10      # depth grows while ADOPTED
LOCK_IN_THRESHOLD = 0.70                # investment_depth above this → LOCKED_IN
BASE_CHURN_RATE = 0.02                  # baseline P(churn) per round while ADOPTED


# ──────────────────────── Phase sync helper ────────────────────────


def sync_aware_flag(agents: np.ndarray) -> None:
    """Re-compute the `aware` boolean from `phase`.

    The forces.py slicing still uses `aware` as its decision-pool mask.
    After any phase transition, call this to keep the two in sync.

    `aware` is True iff phase ∈ DECISION_PHASES (AWARE / CONSIDERING /
    TRIALING). ADOPTED / LOCKED_IN / CHURNED / UNAWARE agents are NOT
    included — they've either decided or don't know about the product.
    """
    phase = agents["phase"]
    agents["aware"] = np.isin(phase, np.array(DECISION_PHASES, dtype=phase.dtype))


# ──────────────────────── Round transitions ────────────────────────


def decay_awareness(
    agents: np.ndarray,
    decay: float = AWARENESS_DECAY_PER_ROUND,
    threshold: float = AWARENESS_REVERT_THRESHOLD,
) -> None:
    """Reduce awareness_strength for agents in decision phases.

    Agents whose strength falls below `threshold` revert to UNAWARE.
    ADOPTED / LOCKED_IN / CHURNED agents don't decay — they've already
    decided and we track their state differently.
    """
    phase = agents["phase"]
    in_decision = np.isin(phase, np.array(DECISION_PHASES, dtype=phase.dtype))
    if not in_decision.any():
        return
    agents["awareness_strength"][in_decision] *= (1.0 - decay)
    # Anyone who dropped below threshold reverts to UNAWARE.
    below = in_decision & (agents["awareness_strength"] < threshold)
    agents["phase"][below] = int(Phase.UNAWARE)
    agents["awareness_strength"][below] = 0.0


def promote_unaware_to_aware(
    agents: np.ndarray,
    product_adoption_rate: float,
    rng: np.random.Generator,
    marketing_reach: float = MARKETING_REACH_PER_ROUND,
    wom_multiplier: float = WOM_MULTIPLIER,
    cluster_rates: np.ndarray | None = None,
) -> int:
    """Promote a subset of UNAWARE agents to AWARE.

    Wave 4: when `cluster_rates` is provided, the WOM term is per-agent:

        p_aware_i = 1 - (1 - marketing_reach)
                      × (1 - cluster_rates[cid_i] * cluster_vis[cid_i]
                             * wom_multiplier)

    Each UNAWARE agent's P(aware) depends on adoption inside their
    cluster, weighted by that cluster's visibility factor. Agents in
    cold clusters stay unaware longer; agents in hot clusters get pulled
    in fast.

    Fallback (`cluster_rates=None`) preserves the Wave 3 global-scalar
    formula for backward compatibility and any direct-call sites.

    Returns the count of agents newly promoted.
    """
    phase = agents["phase"]
    unaware = phase == int(Phase.UNAWARE)
    n_unaware = int(unaware.sum())
    if n_unaware == 0:
        return 0

    if cluster_rates is not None:
        cluster_vis = cluster_visibility_array()
        unaware_cid = agents["cluster_id"][unaware].astype(np.int64)
        wom_term = (
            cluster_rates[unaware_cid]
            * cluster_vis[unaware_cid]
            * wom_multiplier
        )
        p_aware = 1.0 - (1.0 - marketing_reach) * (1.0 - wom_term)
        p_aware = np.clip(p_aware, 0.0, 1.0)
        draws = rng.random(n_unaware)
        promoted_local = draws < p_aware
    else:
        p_aware = 1.0 - (1.0 - marketing_reach) * (
            1.0 - product_adoption_rate * wom_multiplier
        )
        p_aware = float(np.clip(p_aware, 0.0, 1.0))
        draws = rng.random(n_unaware)
        promoted_local = draws < p_aware

    # Map back to global indices.
    global_idx = np.flatnonzero(unaware)
    promoted_global = global_idx[promoted_local]
    agents["phase"][promoted_global] = int(Phase.AWARE)
    agents["awareness_strength"][promoted_global] = INITIAL_AWARENESS_STRENGTH
    return int(promoted_local.sum())


def promote_aware_to_considering(
    agents: np.ndarray,
    params: SimulationParams,
    rng: np.random.Generator,
) -> int:
    """Promote AWARE agents whose awareness_strength crosses the
    consideration threshold (from maturity helper in Wave 1).

    Also adds a small strength gain to simulate "repeated exposure".
    Relevance gate: the product must have at least one positive-polarity
    feature the agent weights meaningfully, else they stay AWARE.
    """
    phase = agents["phase"]
    aware = phase == int(Phase.AWARE)
    n_aware = int(aware.sum())
    if n_aware == 0:
        return 0

    # Each round an aware agent is "re-exposed" and gets a strength bump.
    agents["awareness_strength"][aware] = np.minimum(
        1.0, agents["awareness_strength"][aware] + CONSIDERATION_STRENGTH_GAIN
    )

    threshold = params.consideration_threshold
    crossed = aware & (agents["awareness_strength"] >= threshold)
    n_crossed = int(crossed.sum())
    if n_crossed == 0:
        return 0

    # Relevance: the product has something the agent cares about.
    # For feature-matrix products, relevance = max over positive features of
    # agent_weight[category] × feature.score. For scalar-only products
    # (legacy), every aware agent is relevant by default.
    if params.feature_matrix:
        relevant_agents = agents[crossed]
        gain = compute_feature_gain(
            relevant_agents,
            params.feature_matrix,
            gamma=params.probability_weighting_gamma,
        )
        # Gate: at least some non-trivial gain. Threshold is intentionally
        # low — we're checking "is this product relevant to me at all",
        # not "am I about to adopt".
        relevant = gain > 0.05
    else:
        relevant = np.ones(n_crossed, dtype=bool)

    crossed_idx = np.flatnonzero(crossed)
    to_consider = crossed_idx[relevant]
    agents["phase"][to_consider] = int(Phase.CONSIDERING)
    return len(to_consider)


def decide_considering_to_adopted(
    agents: np.ndarray,
    adopt_prob: np.ndarray,
    rng: np.random.Generator,
    revert_strength_penalty: float = 0.20,
) -> tuple[int, int]:
    """For agents currently CONSIDERING, draw adoption via their P(adopt).

    Agents who adopt → ADOPTED. Agents who *don't* adopt revert to
    AWARE with a penalty on awareness_strength ("I considered it and
    didn't act"). This prevents the multi-round loop from giving every
    considering agent N independent adoption draws — a medium-utility
    agent needs to re-cross the consideration threshold (via re-exposure)
    before they get another chance.

    Returns (n_adopted, n_reverted).
    """
    phase = agents["phase"]
    considering = phase == int(Phase.CONSIDERING)
    n_considering = int(considering.sum())
    if n_considering == 0:
        return 0, 0

    draws = rng.random(n_considering)
    considering_probs = adopt_prob[considering]
    adopting_local = draws < considering_probs
    global_idx = np.flatnonzero(considering)

    adopting_global = global_idx[adopting_local]
    reverting_global = global_idx[~adopting_local]

    # Adopters → ADOPTED with a small investment-depth seed.
    agents["phase"][adopting_global] = int(Phase.ADOPTED)
    agents["investment_depth"][adopting_global] = 0.10

    # Non-adopters → back to AWARE with reduced awareness_strength.
    agents["phase"][reverting_global] = int(Phase.AWARE)
    agents["awareness_strength"][reverting_global] = np.maximum(
        agents["awareness_strength"][reverting_global] - revert_strength_penalty,
        0.0,
    )
    return int(adopting_local.sum()), int((~adopting_local).sum())


def advance_adopted(
    agents: np.ndarray,
    rng: np.random.Generator,
    growth: float = INVESTMENT_GROWTH_PER_ROUND,
    lock_in_threshold: float = LOCK_IN_THRESHOLD,
    base_churn: float = BASE_CHURN_RATE,
) -> tuple[int, int]:
    """Post-decision housekeeping for ADOPTED agents.

    Investment depth grows each round. Deep-enough → LOCKED_IN.
    Otherwise a small churn chance, weighted against investment_depth
    and agent conscientiousness (consistency pressure).

    Returns (n_locked_in, n_churned) for manifest/snapshot reporting.
    """
    phase = agents["phase"]
    adopted = phase == int(Phase.ADOPTED)
    n_adopted = int(adopted.sum())
    if n_adopted == 0:
        return 0, 0

    # Grow investment depth.
    agents["investment_depth"][adopted] = np.minimum(
        1.0, agents["investment_depth"][adopted] + growth
    )

    # Lock-in for deep-invested agents.
    locking = adopted & (agents["investment_depth"] >= lock_in_threshold)
    n_locked = int(locking.sum())
    agents["phase"][locking] = int(Phase.LOCKED_IN)

    # Churn check for the rest. P(churn) scales inversely with depth and
    # with consistency pressure (conscientiousness proxy).
    still_adopted = phase == int(Phase.ADOPTED)  # locking just happened
    n_still = int(still_adopted.sum())
    if n_still == 0:
        return n_locked, 0

    depth = agents["investment_depth"][still_adopted].astype(np.float64)
    conscien = agents["conscientiousness"][still_adopted].astype(np.float64)
    p_churn = base_churn * (1.0 - depth) * (1.0 - 0.5 * conscien)
    p_churn = np.clip(p_churn, 0.0, 1.0)
    draws = rng.random(n_still)
    churning_local = draws < p_churn
    global_idx = np.flatnonzero(still_adopted)
    churning_global = global_idx[churning_local]
    agents["phase"][churning_global] = int(Phase.CHURNED)
    return n_locked, int(churning_local.sum())


# ──────────────────────── Per-round summary ────────────────────────


def phase_counts(agents: np.ndarray) -> dict[str, int]:
    """Return a dict of phase_name → count for the current agent array."""
    phase = agents["phase"]
    out: dict[str, int] = {}
    for p in Phase:
        out[p.name.lower()] = int((phase == int(p)).sum())
    return out


def total_adoption_count(agents: np.ndarray) -> int:
    """Agents currently counted as adopters (ADOPTED + LOCKED_IN)."""
    phase = agents["phase"]
    return int(np.isin(phase, np.array(ADOPTED_PHASES, dtype=phase.dtype)).sum())
