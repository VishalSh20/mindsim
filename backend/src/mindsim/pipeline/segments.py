"""Wave 8 — per-archetype segment narratives.

Pure-Python distillation of `SimulationResult._agents` + `_agent_forces`
into a `SegmentReport` with one `ArchetypeNarrative` per Rogers
archetype. Each narrative carries the archetype's adoption rate, its
dominant driver / blocker (averaged force values across that
archetype's agents), and 3 short representative-agent trajectories.

No LLM calls. The narrative author (A7) consumes the `SegmentReport`
to ground the per-segment paragraph in the report.
"""
from __future__ import annotations

import logging

import numpy as np

from mindsim.engine.archetypes import load_archetypes
from mindsim.models.results import (
    ArchetypeNarrative,
    SegmentReport,
    SimulationResult,
)
from mindsim.models.state import ADOPTED_PHASES, Phase

logger = logging.getLogger(__name__)


# Number of representative agents to sample per archetype.
N_TRAJECTORIES = 3


def build_segment_report(
    sim_result: SimulationResult,
    archetype_names: list[str] | None = None,
) -> SegmentReport:
    """Build per-archetype narratives from a finished SimulationResult.

    Returns an empty report when the result has no `_agents` attached
    (e.g. a fresh-loaded session before snapshot_state runs).
    """
    if archetype_names is None:
        archetype_names = list(load_archetypes().names)

    agents = sim_result._agents
    forces = sim_result._agent_forces
    if agents is None:
        logger.debug("SegmentReport: agents missing; returning empty")
        return SegmentReport()

    narratives: list[ArchetypeNarrative] = []
    for i, aname in enumerate(archetype_names):
        mask = agents["archetype_id"] == i
        total = int(mask.sum())
        if total == 0:
            continue
        narrative = _build_archetype_narrative(
            archetype_name=aname,
            mask=mask,
            agents=agents,
            forces=forces,
            sim_result=sim_result,
        )
        narratives.append(narrative)

    return SegmentReport(segments=narratives)


# ─────────────────────────── per-archetype builder ───────────────────────────


def _build_archetype_narrative(
    archetype_name: str,
    mask: np.ndarray,
    agents: np.ndarray,
    forces: dict[str, np.ndarray] | None,
    sim_result: SimulationResult,
) -> ArchetypeNarrative:
    total = int(mask.sum())

    # Adoption count for this archetype.
    phase_arr = agents["phase"][mask]
    decided = np.isin(
        phase_arr, np.array(ADOPTED_PHASES, dtype=phase_arr.dtype)
    )
    count = int(decided.sum())
    adoption_rate = float(count / total) if total > 0 else 0.0

    driver_name, driver_val, blocker_name, blocker_val = _dominant_forces(
        forces, mask
    )

    trajectories = _representative_trajectories(
        archetype_name=archetype_name,
        mask=mask,
        agents=agents,
        forces=forces,
        probs=sim_result._agent_probs,
    )

    return ArchetypeNarrative(
        archetype=archetype_name,
        adoption_rate=adoption_rate,
        count=count,
        total=total,
        dominant_driver=driver_name,
        dominant_driver_value=driver_val,
        dominant_blocker=blocker_name,
        dominant_blocker_value=blocker_val,
        representative_trajectories=trajectories,
    )


def _dominant_forces(
    forces: dict[str, np.ndarray] | None,
    mask: np.ndarray,
) -> tuple[str, float, str, float]:
    """Compute the per-archetype mean of each force; pick max + min."""
    if forces is None:
        return "", 0.0, "", 0.0

    means: dict[str, float] = {}
    for name, arr in forces.items():
        if arr is None:
            continue
        slice_ = arr[mask]
        valid = slice_[~np.isnan(slice_)]
        if valid.size == 0:
            continue
        means[name] = float(np.mean(valid))

    if not means:
        return "", 0.0, "", 0.0

    driver_name = max(means, key=lambda k: means[k])
    blocker_name = min(means, key=lambda k: means[k])
    return driver_name, means[driver_name], blocker_name, means[blocker_name]


def _representative_trajectories(
    archetype_name: str,
    mask: np.ndarray,
    agents: np.ndarray,
    forces: dict[str, np.ndarray] | None,
    probs,
    n: int = N_TRAJECTORIES,
) -> list[str]:
    """Pick N agents and emit a short prose line per agent."""
    archetype_idx = np.flatnonzero(mask)
    if archetype_idx.size == 0:
        return []

    # Spread the picks: lowest-prob, median-prob, highest-prob agent in the
    # archetype. Surfaces the divergence within a single Rogers segment.
    if probs is not None:
        archetype_probs = np.asarray(probs)[archetype_idx]
        order = np.argsort(archetype_probs)
        if archetype_idx.size <= n:
            picks = archetype_idx[order]
        else:
            picks = archetype_idx[order[[0, len(order) // 2, -1]]]
    else:
        # No probs → just take first N.
        picks = archetype_idx[: min(n, archetype_idx.size)]

    out: list[str] = []
    for pick in picks:
        out.append(_describe_agent(int(pick), agents, forces, probs))
    return out


def _describe_agent(
    idx: int,
    agents: np.ndarray,
    forces: dict[str, np.ndarray] | None,
    probs,
) -> str:
    """One-line plain-English description of agent `idx`'s trajectory."""
    agent = agents[idx]
    phase = int(agent["phase"])
    try:
        phase_label = Phase(phase).name.lower()
    except ValueError:
        phase_label = f"phase_{phase}"

    income = float(agent["income"])
    if income < 30_000:
        income_label = "low-income"
    elif income < 90_000:
        income_label = "mid-income"
    else:
        income_label = "high-income"

    prob_str = ""
    if probs is not None:
        try:
            prob = float(np.asarray(probs)[idx])
            prob_str = f", P(adopt)={prob:.2f}"
        except (IndexError, TypeError):
            pass

    investment = float(agent["investment_depth"])
    aware_strength = float(agent["awareness_strength"])

    # Dominant force for THIS agent (highest absolute value).
    force_phrase = ""
    if forces:
        best_name = ""
        best_abs = -1.0
        for name, arr in forces.items():
            if arr is None:
                continue
            try:
                v = float(arr[idx])
            except (IndexError, TypeError):
                continue
            if np.isnan(v):
                continue
            if abs(v) > best_abs:
                best_abs = abs(v)
                best_name = name
        if best_name:
            sign = "+" if best_abs >= 0 else ""
            force_phrase = f", led by {best_name}"

    return (
        f"#{idx} ({income_label}, awareness {aware_strength:.2f}) "
        f"finished {phase_label} with investment_depth {investment:.2f}"
        f"{prob_str}{force_phrase}."
    )
