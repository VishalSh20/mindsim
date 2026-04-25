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

# Wave 7 — free-trial constants. The trial phase exists in the dtype since
# Wave 3 but had no transitions wired; these constants drive the new
# considering→trialing→adopted/quit edges.
TRIAL_CONSISTENCY_BONUS = 0.15         # Cialdini commitment/consistency lift
TRIAL_SUNK_COST_PER_ROUND = 0.10       # added to conversion utility per trial round invested
TRIAL_INVESTMENT_DEPTH_SEED = 0.30     # converted trial users start with a head start (vs 0.10 for direct)
TRIAL_QUIT_AWARENESS_STRENGTH = 0.30   # quit users remember it, but at reduced strength


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
    free_trial_active: bool = False,
    free_trial_duration_rounds: int = 3,
    free_trial_entry_threshold: float = 0.20,
) -> tuple[int, int, int]:
    """For agents currently CONSIDERING, draw adoption via their P(adopt).

    Agents who adopt → ADOPTED. Agents who *don't* adopt revert to
    AWARE with a penalty on awareness_strength ("I considered it and
    didn't act"). This prevents the multi-round loop from giving every
    considering agent N independent adoption draws — a medium-utility
    agent needs to re-cross the consideration threshold (via re-exposure)
    before they get another chance.

    Wave 7 — when `free_trial_active` is True, considering agents who
    DON'T directly adopt but whose `adopt_prob` clears
    `free_trial_entry_threshold` AND have not previously quit a trial
    (`trial_outcome != 0`) are routed to TRIALING instead of reverting.
    `advance_trialing()` then handles the conversion / quit decision
    `free_trial_duration_rounds` rounds later.

    Returns (n_adopted, n_reverted, n_trialing).
    """
    phase = agents["phase"]
    considering = phase == int(Phase.CONSIDERING)
    n_considering = int(considering.sum())
    if n_considering == 0:
        return 0, 0, 0

    draws = rng.random(n_considering)
    considering_probs = adopt_prob[considering]
    adopting_local = draws < considering_probs
    global_idx = np.flatnonzero(considering)

    adopting_global = global_idx[adopting_local]
    rest_global = global_idx[~adopting_local]

    # Adopters → ADOPTED with a small investment-depth seed.
    agents["phase"][adopting_global] = int(Phase.ADOPTED)
    agents["investment_depth"][adopting_global] = 0.10

    n_trialing = 0
    reverting_global = rest_global

    if free_trial_active and len(rest_global) > 0:
        rest_probs = adopt_prob[rest_global]
        prior_quit = agents["trial_outcome"][rest_global] == 0
        eligible = (rest_probs >= free_trial_entry_threshold) & (~prior_quit)
        trialing_global = rest_global[eligible]
        reverting_global = rest_global[~eligible]
        if len(trialing_global) > 0:
            agents["phase"][trialing_global] = int(Phase.TRIALING)
            agents["trial_rounds_remaining"][trialing_global] = int(
                free_trial_duration_rounds
            )
            agents["trial_outcome"][trialing_global] = -1
            agents["awareness_strength"][trialing_global] = np.maximum(
                agents["awareness_strength"][trialing_global], 0.5
            )
            n_trialing = int(len(trialing_global))

    # Non-adopters who weren't routed to TRIALING → back to AWARE.
    agents["phase"][reverting_global] = int(Phase.AWARE)
    agents["awareness_strength"][reverting_global] = np.maximum(
        agents["awareness_strength"][reverting_global] - revert_strength_penalty,
        0.0,
    )
    return int(adopting_local.sum()), int(len(reverting_global)), n_trialing


def advance_trialing(
    agents: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Tick TRIALING agents one round forward.

    Each round a trial is active, `trial_rounds_remaining` decrements.
    When it reaches 0, the agent decides:

        conversion_score = experienced_value
                         + sunk_cost_bonus(rounds_invested)
                         + TRIAL_CONSISTENCY_BONUS
        switching_drag   = agent_switching_cost
        P(convert) = sigmoid(conversion_score - switching_drag)

    Converters → ADOPTED with a higher `investment_depth` seed than direct
    adopters (they've already invested time). Quitters → AWARE with
    `trial_outcome=0`, blocking re-entry to TRIALING in future rounds.
    `experienced_value` is `agent_perceived_benefit` directly — the trial
    has resolved the certainty the prospect-value pre-trial was discounting.

    Returns (n_converted, n_quit). When called with no TRIALING agents,
    returns (0, 0) — caller can ignore.
    """
    phase = agents["phase"]
    trialing = phase == int(Phase.TRIALING)
    if not trialing.any():
        return 0, 0

    # Decrement rounds-remaining for active trials. Floor at 0 so a finished
    # trial that for any reason wasn't promoted last round doesn't go
    # negative (defensive — current logic always promotes on reach-zero).
    agents["trial_rounds_remaining"][trialing] = np.maximum(
        agents["trial_rounds_remaining"][trialing].astype(np.int16) - 1, 0
    ).astype(np.int8)

    ended = trialing & (agents["trial_rounds_remaining"] <= 0)
    if not ended.any():
        return 0, 0

    ended_idx = np.flatnonzero(ended)
    pb = agents["agent_perceived_benefit"][ended_idx].astype(np.float64)
    sc = agents["agent_switching_cost"][ended_idx].astype(np.float64)

    # Sunk-cost bonus scales with how many rounds the agent invested in
    # the trial. We don't currently store rounds_completed, so use the full
    # trial duration as the upper bound (trial_rounds_remaining hit 0 → all
    # rounds were used). For partial trials (future feature) this would
    # need a separate counter.
    sunk_cost_bonus = TRIAL_SUNK_COST_PER_ROUND * 3.0  # 3 = typical duration

    conversion_score = pb + sunk_cost_bonus + TRIAL_CONSISTENCY_BONUS
    drag = sc
    # Logistic squash. Temperature 2.0 keeps the curve gentle — trial
    # outcomes shouldn't be deterministic on any single feature.
    p_convert = 1.0 / (1.0 + np.exp(-2.0 * (conversion_score - drag)))
    p_convert = np.clip(p_convert, 0.0, 1.0)

    draws = rng.random(len(ended_idx))
    converting_local = draws < p_convert

    converting_global = ended_idx[converting_local]
    quitting_global = ended_idx[~converting_local]

    if len(converting_global) > 0:
        agents["phase"][converting_global] = int(Phase.ADOPTED)
        agents["investment_depth"][converting_global] = TRIAL_INVESTMENT_DEPTH_SEED
        agents["trial_outcome"][converting_global] = 1
        agents["trial_rounds_remaining"][converting_global] = 0

    if len(quitting_global) > 0:
        agents["phase"][quitting_global] = int(Phase.AWARE)
        agents["trial_outcome"][quitting_global] = 0
        agents["awareness_strength"][quitting_global] = TRIAL_QUIT_AWARENESS_STRENGTH
        agents["trial_rounds_remaining"][quitting_global] = 0

    return int(converting_local.sum()), int((~converting_local).sum())


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
