"""Wave 8.5 — `ParameterReasoningTrail` assembly.

Pure-Python composition of `CalibratedParam` + `Assumption` +
`EvidenceStore` into a list of `ParameterTrailEntry` records — one per
headline parameter — with the underlying evidence already resolved.

The narrative author (A7) consumes this list to write the "Evidence &
Reasoning" section. The CLI's `--show-evidence` flag dumps the same
data verbatim.

No LLM calls. Deterministic. Re-runnable on a loaded session.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from mindsim.models.config import (
    Assumption,
    CalibratedParam,
    ReferencePriceParam,
    SimulationConfig,
)
from mindsim.models.evidence import Evidence, EvidenceStore

logger = logging.getLogger(__name__)


class ParameterTrailEntry(BaseModel):
    """One headline parameter with its full evidence chain resolved."""

    parameter: str
    value: float
    basis: str
    confidence: float
    source_type: str | None = None
    published_bounds: tuple[float, float] | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class ParameterReasoningTrail(BaseModel):
    """The full per-parameter reasoning trail for a run."""

    entries: list[ParameterTrailEntry] = Field(default_factory=list)


# Headline parameters surfaced in the report. Other params (population
# config, awareness, etc.) are visible via the diagnostic dump but not
# in the narrative trail.
HEADLINE_PARAMS = (
    "perceived_benefit",
    "benefit_certainty",
    "switching_cost",
    "social_visibility",
    "time_to_value",
    "category_penetration",
    "category_growth",
    "requires_behavior_change",
    "identity_signal",
    "present_bias_beta",
    "fomo_intensity",
    "product_adoption_rate",
)


def build_parameter_reasoning_trail(
    config: SimulationConfig,
    evidence_store: EvidenceStore | None,
) -> ParameterReasoningTrail:
    """Assemble the per-parameter reasoning trail.

    Iterates HEADLINE_PARAMS and the reference price; resolves each
    parameter's `evidence_refs` against `evidence_store`. Assumptions
    that override a headline param fold their `published_bounds` into
    the entry.
    """
    sim = config.simulation_params
    assumption_by_param = _index_assumptions_by_param(config.assumptions)
    entries: list[ParameterTrailEntry] = []

    # Headline 0-1 scalar params.
    for name in HEADLINE_PARAMS:
        cparam = getattr(sim, name, None)
        if not isinstance(cparam, CalibratedParam):
            continue
        entry = _entry_for_param(
            parameter=name,
            cparam=cparam,
            assumption=assumption_by_param.get(name),
            evidence_store=evidence_store,
        )
        entries.append(entry)

    # Reference price gets its own slot (it's a ReferencePriceParam, not
    # a CalibratedParam — different schema, same provenance contract).
    ref = sim.reference_price
    if isinstance(ref, ReferencePriceParam):
        entries.append(_entry_for_reference_price(
            ref,
            assumption=assumption_by_param.get("reference_price"),
            evidence_store=evidence_store,
        ))

    return ParameterReasoningTrail(entries=entries)


# ──────────────────────────── internals ────────────────────────────


def _index_assumptions_by_param(
    assumptions: list[Assumption],
) -> dict[str, Assumption]:
    out: dict[str, Assumption] = {}
    for a in assumptions:
        if a.parameter and a.parameter not in out:
            out[a.parameter] = a
    return out


def _entry_for_param(
    parameter: str,
    cparam: CalibratedParam,
    assumption: Assumption | None,
    evidence_store: EvidenceStore | None,
) -> ParameterTrailEntry:
    refs = list(cparam.evidence_refs or [])
    source_type: str | None = cparam.source_type
    published_bounds: tuple[float, float] | None = None

    if assumption is not None:
        if assumption.evidence_refs:
            # Merge in any extra refs the calibrator emitted on the
            # assumption row (the LLM may attach quotes there).
            for r in assumption.evidence_refs:
                if r not in refs:
                    refs.append(r)
        source_type = source_type or assumption.source_type
        published_bounds = assumption.published_bounds

    evidence = (
        evidence_store.resolve(refs) if evidence_store is not None else []
    )

    return ParameterTrailEntry(
        parameter=parameter,
        value=cparam.value,
        basis=cparam.basis,
        confidence=cparam.confidence,
        source_type=source_type,
        published_bounds=published_bounds,
        evidence=evidence,
    )


def _entry_for_reference_price(
    ref: ReferencePriceParam,
    assumption: Assumption | None,
    evidence_store: EvidenceStore | None,
) -> ParameterTrailEntry:
    refs: list[str] = []
    source_type: str | None = None
    published_bounds: tuple[float, float] | None = None
    if assumption is not None:
        refs = list(assumption.evidence_refs or [])
        source_type = assumption.source_type
        published_bounds = assumption.published_bounds

    evidence = (
        evidence_store.resolve(refs) if evidence_store is not None else []
    )

    return ParameterTrailEntry(
        parameter="reference_price",
        value=ref.value,
        basis=ref.basis,
        confidence=ref.confidence,
        source_type=source_type,
        published_bounds=published_bounds,
        evidence=evidence,
    )


def collect_referenced_evidence_ids(
    config: SimulationConfig,
    pros_cons: list | None = None,
) -> set[str]:
    """Gather every evidence_id cited by any artefact in the run.

    Used by the EvidenceStore compression step in `analyze()`: anything
    NOT in this set is dropped before serialisation.
    """
    referenced: set[str] = set()

    sim = config.simulation_params
    for name in HEADLINE_PARAMS:
        cparam = getattr(sim, name, None)
        if isinstance(cparam, CalibratedParam):
            referenced.update(cparam.evidence_refs or [])

    for a in config.assumptions:
        referenced.update(a.evidence_refs or [])

    if pros_cons:
        for item in pros_cons:
            for q in getattr(item, "voc_evidence", []) or []:
                referenced.add(q)
            sim_ev = getattr(item, "simulation_evidence", {}) or {}
            ref = sim_ev.get("evidence_id")
            if isinstance(ref, str):
                referenced.add(ref)

    return referenced
