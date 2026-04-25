"""Wave 8 — KPI extraction.

Pure-Python distillation of `SimulationResult` + sensitivity into a
publication-ready KPI dashboard. No LLM calls. No RNG. Deterministic
given a `SimulationResult`.

Mines artefacts that prior waves were producing but never surfacing:
  - per-round phase_counts (Wave 3) → time-to-50% + chasm round
  - per-round cluster_adoption (Wave 4) → cascade metrics
  - `_agent_forces` + `_agent_probs` (always populated by simulate()) →
    convertible-pool count + per-archetype force variance

The "convertible pool" count is the strategic lever this dashboard
exists to surface: agents whose adoption probability sits in [0.4, 0.6]
are the ones a small intervention could convert. They're invisible in
the headline adoption number.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from mindsim.models.results import (
    AdoptionSummary,
    CascadeMetrics,
    ForceDominance,
    KPIDashboard,
    SimulationResult,
)

logger = logging.getLogger(__name__)


# Convertible pool window. Wider than the engine's "convertible" proximity
# bucket [0.3, 0.7] — KPI version is the *strategic* window an intervention
# can realistically swing in one go.
CONVERTIBLE_LOW = 0.40
CONVERTIBLE_HIGH = 0.60

# Cluster critical-mass threshold for cascade timing.
CLUSTER_CRITICAL_MASS = 0.50

# Chasm detection: round where adoption growth stalls below this rate
# AFTER the innovator share (~16%) has been reached. Mirrors the Rogers
# diffusion gap between visionaries and pragmatists.
CHASM_GROWTH_THRESHOLD = 0.01  # under 1pp/round counts as a stall
CHASM_LOWER_BOUND = 0.16        # only meaningful past innovator share


def build_kpi_dashboard(
    sim_result: SimulationResult,
    sensitivity_results: Iterable | None = None,
    validation_score: float | None = None,
) -> KPIDashboard:
    """Distill a SimulationResult into a structured KPIDashboard."""
    return KPIDashboard(
        adoption=_adoption_summary(sim_result),
        force_dominance=_force_dominance(sim_result),
        convertible_pool=_convertible_pool_count(sim_result),
        cascade=_cascade_metrics(sim_result),
        top_sensitivity_params=_top_sensitivity_params(sensitivity_results),
        validation_score=validation_score,
    )


# ─────────────────────────── adoption ───────────────────────────


def _adoption_summary(result: SimulationResult) -> AdoptionSummary:
    return AdoptionSummary(
        total=float(result.total_adoption),
        aware=float(result.aware_adoption),
        time_to_50pct=_time_to_50pct(result),
        chasm_round=_chasm_round(result),
    )


def _time_to_50pct(result: SimulationResult) -> int | None:
    """Earliest round where total_adoption ≥ 0.5. None if never reached."""
    for snap in result.rounds:
        if snap.total_adoption >= 0.5:
            return int(snap.round)
    return None


def _chasm_round(result: SimulationResult) -> int | None:
    """First round where adoption growth stalls past the innovator phase.

    The Rogers chasm sits between innovators (~16% combined with early
    adopters) and the early majority. We look for the earliest round
    where adoption is past the innovator share but the round-over-round
    growth fell below the stall threshold — that's where the
    intervention designer needs to focus to "cross the chasm".

    Returns None if growth never stalls in the simulated horizon.
    """
    rounds = result.rounds
    if len(rounds) < 2:
        return None
    for i in range(1, len(rounds)):
        prev = rounds[i - 1].total_adoption
        cur = rounds[i].total_adoption
        if cur < CHASM_LOWER_BOUND:
            continue
        growth = cur - prev
        if growth < CHASM_GROWTH_THRESHOLD:
            return int(rounds[i].round)
    return None


# ─────────────────────────── force dominance ───────────────────────────


def _force_dominance(result: SimulationResult) -> ForceDominance:
    """Top driver, top blocker, and per-archetype variance of each force."""
    forces = result.force_decomposition.as_dict()
    if not forces:
        return ForceDominance()

    top_driver = max(forces.items(), key=lambda kv: kv[1])[0]
    top_blocker = min(forces.items(), key=lambda kv: kv[1])[0]

    segment_variance = _segment_force_variance(result)

    return ForceDominance(
        top_driver=top_driver,
        top_blocker=top_blocker,
        segment_variance=segment_variance,
    )


def _segment_force_variance(result: SimulationResult) -> dict[str, float]:
    """Variance of each force across archetypes.

    Computed from `_agent_forces` (per-agent force values) grouped by
    archetype_id from `_agents`. A high-variance force is one segments
    react to differently — useful signal for "this product polarises".

    Returns {} when per-agent data isn't attached (e.g. on a loaded
    session before `snapshot_state` runs).
    """
    agents = result._agents
    forces = result._agent_forces
    if agents is None or forces is None:
        return {}

    archetype_ids = agents["archetype_id"]
    n_archetypes = int(archetype_ids.max()) + 1 if len(archetype_ids) else 0
    if n_archetypes == 0:
        return {}

    out: dict[str, float] = {}
    for force_name, force_arr in forces.items():
        if force_arr is None:
            continue
        per_archetype_means: list[float] = []
        for a in range(n_archetypes):
            mask = (archetype_ids == a) & ~np.isnan(force_arr)
            if not mask.any():
                continue
            per_archetype_means.append(float(np.nanmean(force_arr[mask])))
        if len(per_archetype_means) >= 2:
            out[force_name] = float(np.var(per_archetype_means))
    return out


# ─────────────────────────── convertible pool ───────────────────────────


def _convertible_pool_count(result: SimulationResult) -> int:
    """Agents whose final-round P(adopt) sits in the [0.4, 0.6] window.

    These are the agents an intervention can realistically flip with a
    small parameter shift. The headline adoption number hides them
    entirely — they look identical to the unreachables.
    """
    probs = result._agent_probs
    if probs is None:
        return 0
    arr = np.asarray(probs)
    if arr.size == 0:
        return 0
    in_window = (arr >= CONVERTIBLE_LOW) & (arr <= CONVERTIBLE_HIGH)
    return int(in_window.sum())


# ─────────────────────────── cascade ───────────────────────────


def _cascade_metrics(result: SimulationResult) -> CascadeMetrics | None:
    """Cluster-by-cluster spread timing.

    For each cluster, finds the earliest round where its in-cluster
    adoption rate crosses CLUSTER_CRITICAL_MASS. Returns None when no
    cluster-level data was recorded (Wave 3 single-cluster path).
    """
    rounds = result.rounds
    if not rounds:
        return None

    # Collect the set of cluster ids that ever appear.
    cluster_ids: set[int] = set()
    for snap in rounds:
        cluster_ids.update(int(c) for c in snap.cluster_adoption.keys())
    if not cluster_ids:
        return None

    spread_rounds: dict[int, int] = {}
    for cid in sorted(cluster_ids):
        for snap in rounds:
            rate = snap.cluster_adoption.get(cid)
            if rate is None:
                continue
            if rate >= CLUSTER_CRITICAL_MASS:
                spread_rounds[int(cid)] = int(snap.round)
                break

    first_cluster = (
        min(spread_rounds, key=lambda c: spread_rounds[c]) if spread_rounds else None
    )

    return CascadeMetrics(
        first_cluster_crossed_critical_mass=first_cluster,
        cluster_spread_rounds=spread_rounds,
    )


# ─────────────────────────── sensitivity ───────────────────────────


def _top_sensitivity_params(sensitivity_results: Iterable | None) -> list[str]:
    """Top-3 parameter names by swing magnitude."""
    if sensitivity_results is None:
        return []
    items = list(sensitivity_results)
    if not items:
        return []
    items.sort(key=lambda s: getattr(s, "swing", 0.0), reverse=True)
    return [getattr(s, "parameter", "") for s in items[:3] if getattr(s, "parameter", "")]
