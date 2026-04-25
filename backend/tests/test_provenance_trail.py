"""Wave 8.5 — ParameterReasoningTrail builder tests."""
from __future__ import annotations

import pytest

from mindsim.models.config import (
    Assumption,
    CalibratedParam,
    PopulationConfig,
    ReferencePriceParam,
    SimulationConfig,
    SimulationParams,
)
from mindsim.models.evidence import Evidence, EvidenceStore
from mindsim.pipeline.provenance import (
    HEADLINE_PARAMS,
    build_parameter_reasoning_trail,
    collect_referenced_evidence_ids,
)


def _config_with_evidence() -> tuple[SimulationConfig, EvidenceStore]:
    store = EvidenceStore()
    store.add(Evidence(id="voc:abc", text="users love it", source_type="voc"))
    store.add(Evidence(id="src:def", text="market data", source_type="research"))

    params = SimulationParams(
        price=20.0,
        reference_price=ReferencePriceParam(
            value=18.0, basis="median competitor", confidence=0.7,
        ),
        category_penetration=CalibratedParam(
            value=0.30,
            basis="industry report",
            confidence=0.7,
            evidence_refs=["src:def"],
            source_type="research",
        ),
        present_bias_beta=CalibratedParam(
            value=0.55,
            basis="Augenblick 2015 default",
            confidence=0.5,
            evidence_refs=[],
            source_type="default",
        ),
        perceived_benefit=CalibratedParam(
            value=0.65,
            basis="VoC delight quotes",
            confidence=0.6,
            evidence_refs=["voc:abc"],
            source_type="voc",
        ),
    )
    assumptions = [
        Assumption(
            id="A1",
            parameter="present_bias_beta",
            value=0.55,
            basis="Augenblick 2015",
            confidence=0.5,
            sensitivity="high",
            published_bounds=(0.3, 0.95),
            source_type="default",
        ),
    ]
    return SimulationConfig(
        simulation_params=params,
        population_config=PopulationConfig(),
        assumptions=assumptions,
    ), store


class TestTrailBuilder:
    def test_emits_one_entry_per_headline_param_plus_reference_price(self):
        config, store = _config_with_evidence()
        trail = build_parameter_reasoning_trail(config, store)
        param_names = [e.parameter for e in trail.entries]
        assert "reference_price" in param_names
        for hp in HEADLINE_PARAMS:
            assert hp in param_names

    def test_evidence_resolved_when_refs_match_store(self):
        config, store = _config_with_evidence()
        trail = build_parameter_reasoning_trail(config, store)
        pen_entry = next(e for e in trail.entries if e.parameter == "category_penetration")
        assert len(pen_entry.evidence) == 1
        assert pen_entry.evidence[0].id == "src:def"

        pb_entry = next(e for e in trail.entries if e.parameter == "perceived_benefit")
        assert pb_entry.evidence[0].id == "voc:abc"

    def test_unknown_refs_dropped_silently(self):
        config, store = _config_with_evidence()
        # Add a fabricated ref not in the store.
        config.simulation_params.category_penetration.evidence_refs.append(
            "voc:NOT_REAL"
        )
        trail = build_parameter_reasoning_trail(config, store)
        pen_entry = next(e for e in trail.entries if e.parameter == "category_penetration")
        # Resolve dropped the unknown; only the real one survives.
        assert [ev.id for ev in pen_entry.evidence] == ["src:def"]

    def test_assumption_published_bounds_attach_to_entry(self):
        config, store = _config_with_evidence()
        trail = build_parameter_reasoning_trail(config, store)
        pb_entry = next(e for e in trail.entries if e.parameter == "present_bias_beta")
        assert pb_entry.published_bounds == (0.3, 0.95)
        assert pb_entry.source_type == "default"

    def test_missing_evidence_store_returns_empty_evidence_lists(self):
        config, _ = _config_with_evidence()
        trail = build_parameter_reasoning_trail(config, evidence_store=None)
        for entry in trail.entries:
            assert entry.evidence == []


class TestCollectReferenced:
    def test_gathers_param_assumption_and_pros_cons_refs(self):
        from mindsim.models.results import EvidenceStrength, ProsConsItem

        config, _ = _config_with_evidence()
        config.simulation_params.perceived_benefit.evidence_refs = ["voc:p1"]
        config.assumptions[0].evidence_refs = ["src:a1"]
        pros_cons = [
            ProsConsItem(
                statement="x", polarity="con",
                voc_evidence=["voc:pc1", "voc:pc2"],
                simulation_evidence={"evidence_id": "src:pc"},
                evidence_strength=EvidenceStrength(),
            ),
        ]
        ids = collect_referenced_evidence_ids(config, pros_cons)
        assert "voc:p1" in ids
        assert "src:a1" in ids
        assert "voc:pc1" in ids and "voc:pc2" in ids
        assert "src:pc" in ids

    def test_works_with_no_pros_cons(self):
        config, _ = _config_with_evidence()
        ids = collect_referenced_evidence_ids(config, pros_cons=None)
        assert "src:def" in ids
