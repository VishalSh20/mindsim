"""Tests for the A5 Validator — hard-range checks.

The LLM cross-field review half is wired in Wave 2 Commit 2. Commit 1
ships only the pure-Python range checks.
"""
import pytest

from mindsim.models.config import (
    CalibratedParam,
    ReferencePriceParam,
    SimulationConfig,
    SimulationParams,
)
from mindsim.models.product import Feature, ProductProfile
from mindsim.pipeline.orchestrator import StageContext
from mindsim.pipeline.validator import (
    FEATURE_COUNT_MAX,
    FEATURE_COUNT_MIN,
    GAMMA_LOCK,
    Validator,
    ValidatorInput,
    check_config_bounds,
    check_feature_matrix,
)


def _valid_feature(name: str = "x") -> Feature:
    return Feature(
        name=name,
        score=0.5,
        polarity="positive",
        category="core_value",
        certainty=0.6,
        visibility=0.3,
        time_to_value_months=1.0,
    )


def _valid_config(features: list[Feature] | None = None) -> SimulationConfig:
    params = SimulationParams(
        price=20.0,
        reference_price=ReferencePriceParam(value=25.0),
        feature_matrix=features if features is not None else [
            _valid_feature(f"feat_{i}") for i in range(5)
        ],
    )
    return SimulationConfig(simulation_params=params)


class TestFeatureMatrixChecks:
    def test_valid_matrix_produces_no_issues(self):
        features = [_valid_feature(f"f_{i}") for i in range(5)]
        assert check_feature_matrix(features) == []

    def test_too_few_features_flagged(self):
        features = [_valid_feature(f"f_{i}") for i in range(FEATURE_COUNT_MIN - 1)]
        issues = check_feature_matrix(features)
        assert any("too few" in i.message for i in issues)

    def test_too_many_features_flagged(self):
        features = [_valid_feature(f"f_{i}") for i in range(FEATURE_COUNT_MAX + 1)]
        issues = check_feature_matrix(features)
        assert any("too many" in i.message for i in issues)

    def test_out_of_range_score(self):
        bad = _valid_feature()
        bad.score = 1.5
        features = [bad] + [_valid_feature(f"f_{i}") for i in range(4)]
        issues = check_feature_matrix(features)
        assert any("score" in i.field for i in issues)

    def test_out_of_range_certainty(self):
        bad = _valid_feature()
        bad.certainty = -0.2
        features = [bad] + [_valid_feature(f"f_{i}") for i in range(4)]
        issues = check_feature_matrix(features)
        assert any("certainty" in i.field for i in issues)

    def test_time_to_value_over_ceiling(self):
        bad = _valid_feature()
        bad.time_to_value_months = 60.0
        features = [bad] + [_valid_feature(f"f_{i}") for i in range(4)]
        issues = check_feature_matrix(features)
        assert any("time_to_value" in i.field for i in issues)


class TestConfigBoundsChecks:
    def test_valid_config_no_issues(self):
        assert check_config_bounds(_valid_config()) == []

    def test_out_of_bounds_beta(self):
        cfg = _valid_config()
        cfg.simulation_params.present_bias_beta = CalibratedParam(value=0.1)
        issues = check_config_bounds(cfg)
        assert any("present_bias_beta" in i.field for i in issues)

    def test_gamma_lock_enforced(self):
        cfg = _valid_config()
        cfg.simulation_params.probability_weighting_gamma = 0.5
        issues = check_config_bounds(cfg)
        assert any("probability_weighting_gamma" in i.field for i in issues)

    def test_gamma_unchanged_passes(self):
        cfg = _valid_config()
        assert cfg.simulation_params.probability_weighting_gamma == GAMMA_LOCK
        assert check_config_bounds(cfg) == []

    def test_penetration_out_of_range(self):
        cfg = _valid_config()
        cfg.simulation_params.category_penetration = CalibratedParam(value=1.5)
        issues = check_config_bounds(cfg)
        assert any("category_penetration" in i.field for i in issues)


class TestValidatorStage:
    def test_validator_passes_valid_bundle(self):
        v = Validator()
        inp = ValidatorInput(
            product=ProductProfile(name="X", raw_description="x"),
            config=_valid_config(),
        )
        report = v.run(inp, StageContext())
        assert report.is_valid is True
        assert report.issues == []

    def test_validator_fails_on_bad_matrix(self):
        v = Validator()
        inp = ValidatorInput(
            product=ProductProfile(name="X", raw_description="x"),
            config=_valid_config(features=[_valid_feature("only_one")]),
        )
        report = v.run(inp, StageContext())
        assert report.is_valid is False
        assert any("too few" in i.message for i in report.issues)

    def test_validator_satisfies_stage_protocol(self):
        from mindsim.pipeline.orchestrator import Stage
        v = Validator()
        assert isinstance(v, Stage)

    def test_validator_returns_validation_report_type(self):
        v = Validator()
        assert v.output_type.__name__ == "ValidationReport"
