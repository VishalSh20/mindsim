"""Construction tests for the Pydantic models added in Wave 0.

These models are not yet wired into the pipeline; they exist so later waves
can import and populate them without schema churn.
"""
import pytest
from pydantic import ValidationError

from mindsim.models.results import (
    AdoptionSummary,
    CascadeMetrics,
    ForceDominance,
    KPIDashboard,
    ProsConsItem,
    RoundSnapshot,
)
from mindsim.models.voc import FeatureSentiment, VoCReport


def test_voc_report_defaults():
    report = VoCReport()
    assert report.pain_points == []
    assert report.bias_note is None
    assert report.source_count == 0


def test_voc_report_with_bias_note():
    report = VoCReport(
        pain_points=["slow startup"],
        bias_note="weighted toward innovators/early adopters",
        source_count=42,
    )
    assert "innovators" in report.bias_note


def test_feature_sentiment_required_fields():
    fs = FeatureSentiment(feature="pricing", positive_pct=0.3, negative_pct=0.6)
    assert fs.quote_ids == []


def test_round_snapshot_construction():
    rs = RoundSnapshot(round=1, total_adoption=0.05, aware_count=100)
    assert rs.phase_counts == {}
    assert rs.cluster_adoption == {}


def test_round_snapshot_with_data():
    rs = RoundSnapshot(
        round=3,
        phase_counts={"unaware": 500, "aware": 300, "considering": 100, "adopted": 100},
        cluster_adoption={0: 0.2, 1: 0.05},
        total_adoption=0.10,
        aware_count=500,
    )
    assert rs.phase_counts["adopted"] == 100
    assert rs.cluster_adoption[0] == 0.2


def test_pros_cons_item_polarity():
    pro = ProsConsItem(statement="cheap", polarity="pro", segments=["innovator"])
    assert pro.polarity == "pro"
    assert pro.voc_evidence == []


def test_pros_cons_rejects_bad_polarity():
    with pytest.raises(ValidationError):
        ProsConsItem(statement="x", polarity="maybe")  # type: ignore[arg-type]


def test_kpi_dashboard_construction():
    kpi = KPIDashboard(
        adoption=AdoptionSummary(total=0.2, aware=0.5),
        force_dominance=ForceDominance(top_driver="prospect", top_blocker="status_quo"),
        convertible_pool=120,
    )
    assert kpi.validation_score is None
    assert kpi.cascade is None
    assert kpi.top_sensitivity_params == []


def test_kpi_dashboard_with_cascade():
    kpi = KPIDashboard(
        cascade=CascadeMetrics(
            first_cluster_crossed_critical_mass=2,
            cluster_spread_rounds={0: 1, 1: 3, 2: 2},
        )
    )
    assert kpi.cascade.first_cluster_crossed_critical_mass == 2
