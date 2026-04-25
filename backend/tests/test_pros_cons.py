"""Wave 8 — pros/cons triangulation enforcer tests."""
from __future__ import annotations

import pytest

from mindsim.models.results import (
    ForceDecomposition,
    ProsConsItem,
    SimulationResult,
)
from mindsim.models.voc import FeatureSentiment, VoCQuote, VoCReport
from mindsim.pipeline.pros_cons import build_pros_cons


def _result_with_forces(**forces) -> SimulationResult:
    return SimulationResult(
        total_adoption=0.5,
        force_decomposition=ForceDecomposition(**forces),
    )


# ─────────────────── triangulation succeeds ───────────────────


class TestTriangulationSucceeds:
    def test_blocker_with_matching_voc_pain_emits_triangulated_con(self):
        # status_quo is most negative; pain_point keyword "switch" matches
        # FORCE_KEYWORDS["status_quo"] → triangulated con.
        result = _result_with_forces(
            prospect_value=0.30,
            anchoring=-0.05,
            status_quo=-0.40,  # top blocker
            social_proof=0.10,
            fomo=0.05,
            hyperbolic_discounting=-0.10,
            identity_signaling=0.02,
        )
        voc = VoCReport(
            pain_points=["users say it's hard to switch from their current tool"],
            delight_points=[],
        )
        items = build_pros_cons(result, voc)
        cons = [i for i in items if i.polarity == "con"]
        assert len(cons) >= 1
        assert any("status_quo" in c.mechanism for c in cons)
        # Triangulated con must reference VoC pain in the statement.
        assert any("switch" in c.statement.lower() for c in cons)

    def test_driver_with_matching_voc_delight_emits_triangulated_pro(self):
        result = _result_with_forces(
            prospect_value=0.50,  # top driver
            anchoring=-0.05,
            status_quo=-0.10,
            social_proof=0.20,
            fomo=0.05,
            hyperbolic_discounting=-0.05,
            identity_signaling=0.02,
        )
        voc = VoCReport(
            pain_points=[],
            delight_points=["everyone says the value is great"],
        )
        items = build_pros_cons(result, voc)
        pros = [i for i in items if i.polarity == "pro"]
        assert len(pros) >= 1
        assert any("prospect_value" in p.mechanism for p in pros)


# ─────────────────── single-source nuance routing ───────────────────


class TestNuanceRouting:
    def test_sim_blocker_without_voc_match_becomes_nuance(self):
        # status_quo is the top blocker but VoC has no matching pain.
        result = _result_with_forces(
            prospect_value=0.30,
            status_quo=-0.40,
            anchoring=-0.05,
            social_proof=0.10,
            fomo=0.0,
            hyperbolic_discounting=0.0,
            identity_signaling=0.0,
        )
        voc = VoCReport(pain_points=["unrelated pain about server downtime"])
        items = build_pros_cons(result, voc)
        # Sim-only nuance is present.
        assert any(
            i.polarity == "nuance" and i.mechanism == "simulation only"
            for i in items
        )
        # No false-positive "con" — status_quo blocker without VoC support.
        cons = [i for i in items if i.polarity == "con"]
        assert not any("status_quo" in c.mechanism for c in cons)

    def test_voc_pain_without_sim_match_becomes_nuance(self):
        # No matching force keyword → VoC-only nuance.
        result = _result_with_forces(
            prospect_value=0.30,
            status_quo=-0.10,
            anchoring=-0.05,
            social_proof=0.10,
            fomo=0.0,
            hyperbolic_discounting=0.0,
            identity_signaling=0.0,
        )
        voc = VoCReport(pain_points=["the API endpoints respond slowly sometimes"])
        items = build_pros_cons(result, voc)
        voc_only = [
            i for i in items
            if i.polarity == "nuance" and i.mechanism == "VoC only"
        ]
        assert len(voc_only) >= 1
        assert any(
            "API endpoints respond slowly" in i.statement for i in voc_only
        )

    def test_empty_voc_collapses_every_sim_signal_to_nuance(self):
        result = _result_with_forces(
            prospect_value=0.30,
            status_quo=-0.40,
            anchoring=-0.05,
            social_proof=0.10,
            fomo=0.0,
            hyperbolic_discounting=0.0,
            identity_signaling=0.0,
        )
        items = build_pros_cons(result, voc=VoCReport.empty())
        triangulated = [i for i in items if i.polarity in ("pro", "con")]
        nuances = [i for i in items if i.polarity == "nuance"]
        # No triangulation possible without VoC themes.
        assert not triangulated
        assert nuances


# ─────────────────── voc-evidence wiring ───────────────────


class TestVocEvidence:
    def test_feature_sentiment_quote_ids_attach_when_keyword_matches(self):
        # status_quo blocker + feature_sentiment whose feature name
        # contains a status_quo keyword ("switch") → quote_ids carry through.
        result = _result_with_forces(
            prospect_value=0.30,
            status_quo=-0.40,
            anchoring=-0.05,
            social_proof=0.10,
            fomo=0.0,
            hyperbolic_discounting=0.0,
            identity_signaling=0.0,
        )
        voc = VoCReport(
            pain_points=["users get stuck and find it hard to switch"],
            delight_points=[],
            feature_sentiment=[
                FeatureSentiment(
                    feature="switching pain",
                    positive_pct=0.05,
                    negative_pct=0.70,
                    quote_ids=["voc:abc123", "voc:def456"],
                )
            ],
        )
        items = build_pros_cons(result, voc)
        cons = [i for i in items if i.polarity == "con"]
        assert cons
        assert cons[0].voc_evidence == ["voc:abc123", "voc:def456"]


# ─────────────────── ordering invariant ───────────────────


class TestOrdering:
    def test_triangulated_items_precede_nuances(self):
        result = _result_with_forces(
            prospect_value=0.50,  # top driver
            status_quo=-0.40,     # top blocker
            anchoring=-0.05,
            social_proof=0.10,
            fomo=0.0,
            hyperbolic_discounting=0.0,
            identity_signaling=0.0,
        )
        voc = VoCReport(
            pain_points=["it's hard to switch tools"],
            delight_points=["the value is huge"],
        )
        items = build_pros_cons(result, voc)
        # Find the index of the first nuance.
        first_nuance_idx = next(
            (i for i, p in enumerate(items) if p.polarity == "nuance"),
            None,
        )
        if first_nuance_idx is not None:
            triangulated_after_nuance = [
                p for p in items[first_nuance_idx:]
                if p.polarity in ("pro", "con")
            ]
            assert not triangulated_after_nuance


# ─────────────────── tolerates None / missing voc ───────────────────


class TestNoneVoc:
    def test_none_voc_does_not_raise(self):
        result = _result_with_forces(prospect_value=0.30, status_quo=-0.20)
        items = build_pros_cons(result, voc=None)
        # Still emits the sim-only nuances; no triangulated rows.
        assert all(i.polarity == "nuance" for i in items)
