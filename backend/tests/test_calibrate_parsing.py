"""Tests for Wave 2 calibrate parsing.

Directly exercises `_parse_calibration_response` — no LLM required.
Closes the "no tests for calibrate parsing" gap from ISSUES.md.
"""
import pytest

from mindsim.models.config import ARCHETYPE_NAMES, SimulationConfig
from mindsim.models.product import ProductProfile
from mindsim.pipeline.calibrate import (
    DEFAULT_FEATURE_MATRIX,
    _derive_aggregate_scalars,
    _parse_calibration_response,
    _parse_competitor_feature_scores,
    _parse_feature_matrix,
)


def _profile() -> ProductProfile:
    return ProductProfile(
        name="Test",
        raw_description="A test product",
        price=20.0,
    )


def _good_llm_response() -> dict:
    return {
        "price": 20.0,
        "reference_price": {
            "value": 25.0,
            "components": [{"source": "Competitor A", "price": 30.0, "weight": 0.6}],
            "confidence": 0.7,
            "basis": "competitor-blend",
        },
        "feature_matrix": [
            {"name": "quality", "polarity": "positive", "category": "core_value",
             "score": 0.8, "certainty": 0.7, "visibility": 0.2, "time_to_value_months": 0.5, "basis": "x"},
            {"name": "cost_tax", "polarity": "negative", "category": "ongoing_cost",
             "score": 0.4, "certainty": 0.8, "visibility": 0.1, "time_to_value_months": 0.0, "basis": "x"},
            {"name": "integrations", "polarity": "positive", "category": "switching_friction_reducer",
             "score": 0.5, "certainty": 0.6, "visibility": 0.2, "time_to_value_months": 0.0, "basis": "x"},
            {"name": "signal", "polarity": "positive", "category": "social_signal",
             "score": 0.3, "certainty": 0.5, "visibility": 0.6, "time_to_value_months": 1.0, "basis": "x"},
        ],
        "competitor_feature_scores": {
            "Competitor A": {"quality": 0.6, "cost_tax": 0.5},
        },
        "category_penetration": {"value": 0.15, "basis": "x", "confidence": 0.5},
        "category_growth": {"value": 0.8, "basis": "x", "confidence": 0.5},
        "requires_behavior_change": {"value": 0.3, "basis": "x", "confidence": 0.5},
        "identity_signal": {"value": 0.3, "basis": "x", "confidence": 0.5},
        "present_bias_beta": {"value": 0.75, "basis": "x", "confidence": 0.6},
        "fomo_intensity": {"value": 0.4, "basis": "x", "confidence": 0.5},
        "product_adoption_rate": {"value": 0.05, "basis": "x", "confidence": 0.5},
        "awareness_by_archetype": {
            "innovator": 0.9, "early_adopter": 0.7, "early_majority": 0.3,
            "late_majority": 0.1, "laggard": 0.02,
        },
        "population_config": {"income_mean_log": 11.0, "income_sigma": 0.7, "market_segment": "general"},
        "assumptions": [
            {"parameter": "category_growth", "value": 0.8, "basis": "x", "confidence": 0.7, "sensitivity": "medium"}
        ],
    }


class TestFeatureMatrixParsing:
    def test_parses_valid_matrix(self):
        features = _parse_feature_matrix(_good_llm_response())
        assert len(features) == 4
        assert features[0].name == "quality"
        assert features[1].polarity == "negative"

    def test_empty_matrix_falls_back_to_default(self):
        features = _parse_feature_matrix({"feature_matrix": []})
        assert len(features) == len(DEFAULT_FEATURE_MATRIX)

    def test_missing_matrix_falls_back_to_default(self):
        features = _parse_feature_matrix({})
        assert len(features) == len(DEFAULT_FEATURE_MATRIX)

    def test_malformed_features_skipped(self):
        response = {
            "feature_matrix": [
                {"name": "good", "polarity": "positive", "category": "core_value",
                 "score": 0.5, "certainty": 0.5, "visibility": 0.3, "time_to_value_months": 0.0, "basis": ""},
                "not a dict",
                {"polarity": "invalid_polarity"},  # will raise in Pydantic
            ]
        }
        features = _parse_feature_matrix(response)
        assert len(features) == 1
        assert features[0].name == "good"

    def test_non_list_matrix_falls_back(self):
        features = _parse_feature_matrix({"feature_matrix": "nope"})
        assert len(features) == len(DEFAULT_FEATURE_MATRIX)


class TestCompetitorFeatureScores:
    def test_parses_nested_dict(self):
        scores = _parse_competitor_feature_scores(_good_llm_response())
        assert "Competitor A" in scores
        assert scores["Competitor A"]["quality"] == 0.6

    def test_rejects_non_dict_values(self):
        raw = {"competitor_feature_scores": {"A": "not a dict"}}
        assert _parse_competitor_feature_scores(raw) == {}

    def test_strips_non_numeric_scores(self):
        raw = {"competitor_feature_scores": {"A": {"ok": 0.5, "bad": "nope"}}}
        scores = _parse_competitor_feature_scores(raw)
        assert scores == {"A": {"ok": 0.5}}


class TestDerivedAggregateScalars:
    def test_all_five_scalars_produced(self):
        features = _parse_feature_matrix(_good_llm_response())
        derived = _derive_aggregate_scalars(features)
        assert set(derived.keys()) == {
            "perceived_benefit", "benefit_certainty", "switching_cost",
            "social_visibility", "time_to_value",
        }

    def test_base_is_early_majority(self):
        features = _parse_feature_matrix(_good_llm_response())
        derived = _derive_aggregate_scalars(features)
        # Base value + by_archetype overrides for 4 other archetypes.
        for field, param in derived.items():
            assert param.by_archetype is not None
            assert set(param.by_archetype.keys()) == set(ARCHETYPE_NAMES) - {"early_majority"}

    def test_perceived_benefit_higher_for_innovators(self):
        """Innovators weight core_value higher → higher perceived_benefit
        for a product that's strong on core_value."""
        # Build a product that's strong on core_value only.
        from mindsim.models.product import Feature
        features = [
            Feature(name="strong_core", polarity="positive", category="core_value",
                    score=0.9, certainty=0.9, visibility=0.2, time_to_value_months=0.0, basis=""),
            Feature(name="weak_cost", polarity="negative", category="ongoing_cost",
                    score=0.2, certainty=0.8, visibility=0.1, time_to_value_months=0.0, basis=""),
            Feature(name="weak_sig", polarity="positive", category="social_signal",
                    score=0.2, certainty=0.5, visibility=0.3, time_to_value_months=0.0, basis=""),
            Feature(name="integ", polarity="positive", category="switching_friction_reducer",
                    score=0.3, certainty=0.6, visibility=0.1, time_to_value_months=0.0, basis=""),
        ]
        derived = _derive_aggregate_scalars(features)
        pb = derived["perceived_benefit"]
        # Innovator weights core_value at 0.50 vs laggard 0.20 → innovator sees higher pb.
        assert pb.by_archetype["innovator"] > pb.by_archetype["laggard"]

    def test_time_to_value_normalised_to_unit_interval(self):
        from mindsim.models.product import Feature
        # Construct a feature with extreme time_to_value.
        features = [
            Feature(name="slow", polarity="positive", category="core_value",
                    score=0.5, certainty=0.5, visibility=0.2,
                    time_to_value_months=24.0, basis=""),
        ] + list(DEFAULT_FEATURE_MATRIX[1:])  # pad to 4 features
        derived = _derive_aggregate_scalars(features)
        ttv = derived["time_to_value"]
        assert 0.0 <= ttv.value <= 1.0

    def test_empty_features_returns_fallback_values(self):
        derived = _derive_aggregate_scalars([])
        # Returns defaults rather than crashing.
        assert 0.0 <= derived["perceived_benefit"].value <= 1.0


class TestFullParse:
    def test_complete_response_builds_config(self):
        config = _parse_calibration_response(_good_llm_response(), _profile())
        assert isinstance(config, SimulationConfig)
        p = config.simulation_params
        assert p.price == 20.0
        assert p.reference_price.value == 25.0
        assert len(p.feature_matrix) == 4
        assert p.competitor_feature_scores["Competitor A"]["quality"] == 0.6

    def test_missing_fields_use_defaults(self):
        minimal = {"feature_matrix": _good_llm_response()["feature_matrix"]}
        config = _parse_calibration_response(minimal, _profile())
        # Should not raise; defaults fill in.
        assert isinstance(config, SimulationConfig)
        assert config.simulation_params.present_bias_beta.value == 0.75

    def test_derived_scalars_populated_even_without_llm_emitting_them(self):
        config = _parse_calibration_response(_good_llm_response(), _profile())
        # perceived_benefit etc. come from derivation.
        assert "derived from feature_matrix" in config.simulation_params.perceived_benefit.basis

    def test_skip_research_safe(self):
        """Test calibration still works with just a feature_matrix field."""
        minimal = {
            "feature_matrix": _good_llm_response()["feature_matrix"],
            "price": 10.0,
        }
        config = _parse_calibration_response(minimal, _profile())
        assert len(config.simulation_params.feature_matrix) == 4
