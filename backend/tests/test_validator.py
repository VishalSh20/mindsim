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
        # Wave 8.5: a default config has empty `basis` strings on every
        # CalibratedParam → sparse-rationale WARNINGS fire. Validity is
        # determined by error-severity issues only, so the bundle still
        # passes; warnings are informational.
        assert report.is_valid is True
        errors = [i for i in report.issues if i.severity == "error"]
        assert errors == []

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


# ───────────────────── Wave 8.5 — provenance + cross-field ─────────────────────


class TestEvidenceRefValidation:
    def _config_with_param_ref(self, ref: str) -> SimulationConfig:
        config = _valid_config()
        config.simulation_params.perceived_benefit = CalibratedParam(
            value=0.6,
            basis="VoC corroboration",
            confidence=0.7,
            evidence_refs=[ref],
            source_type="voc",
        )
        return config

    def test_unresolved_evidence_ref_emits_error(self):
        from mindsim.models.evidence import Evidence, EvidenceStore

        config = self._config_with_param_ref("voc:FAKE")
        store = EvidenceStore()
        store.add(Evidence(id="voc:REAL", text="x", source_type="voc"))

        v = Validator(evidence_store=store)
        report = v.run(ValidatorInput(
            product=ProductProfile(name="x"), config=config,
        ), ctx=StageContext())
        assert not report.is_valid
        assert any(
            "voc:FAKE" in i.message and i.severity == "error"
            for i in report.issues
        )

    def test_resolved_evidence_ref_passes(self):
        from mindsim.models.evidence import Evidence, EvidenceStore

        config = self._config_with_param_ref("voc:REAL")
        store = EvidenceStore()
        store.add(Evidence(id="voc:REAL", text="x", source_type="voc"))

        v = Validator(evidence_store=store)
        report = v.run(ValidatorInput(
            product=ProductProfile(name="x"), config=config,
        ), ctx=StageContext())
        # No error for evidence_refs resolution (other range checks may fire,
        # but no evidence-ref error specifically).
        assert not any(
            "evidence_ref" in i.field for i in report.issues
            if i.severity == "error"
        )

    def test_no_store_skips_resolution_check(self):
        config = self._config_with_param_ref("voc:NOT_RESOLVABLE_BUT_NO_STORE")
        v = Validator(evidence_store=None)
        report = v.run(ValidatorInput(
            product=ProductProfile(name="x"), config=config,
        ), ctx=StageContext())
        # Resolution check skipped when store is None.
        assert not any("does not resolve" in i.message for i in report.issues)


class TestSparseRationale:
    def test_thin_basis_with_no_refs_emits_warning(self):
        config = _valid_config()
        config.simulation_params.perceived_benefit = CalibratedParam(
            value=0.6, basis="x", confidence=0.7,  # 1-char basis
            evidence_refs=[], source_type=None,
        )
        v = Validator()
        report = v.run(ValidatorInput(
            product=ProductProfile(name="x"), config=config,
        ), ctx=StageContext())
        warnings = [i for i in report.issues if i.severity == "warning"]
        assert any(
            "sparse rationale" in w.message and "perceived_benefit" in w.field
            for w in warnings
        )

    def test_default_source_type_opts_out_of_sparse_check(self):
        config = _valid_config()
        config.simulation_params.perceived_benefit = CalibratedParam(
            value=0.6, basis="x", confidence=0.7,
            evidence_refs=[], source_type="default",
        )
        v = Validator()
        report = v.run(ValidatorInput(
            product=ProductProfile(name="x"), config=config,
        ), ctx=StageContext())
        assert not any(
            "sparse rationale" in i.message and "perceived_benefit" in i.field
            for i in report.issues
        )

    def test_thick_basis_passes(self):
        config = _valid_config()
        config.simulation_params.perceived_benefit = CalibratedParam(
            value=0.6,
            basis="industry report cites 60-70% feature satisfaction",
            confidence=0.7,
            evidence_refs=[], source_type=None,
        )
        v = Validator()
        report = v.run(ValidatorInput(
            product=ProductProfile(name="x"), config=config,
        ), ctx=StageContext())
        assert not any(
            "sparse rationale" in i.message and "perceived_benefit" in i.field
            for i in report.issues
        )


class TestCrossField:
    def test_high_penetration_high_growth_inconsistency_warns(self):
        config = _valid_config()
        config.simulation_params.category_penetration = CalibratedParam(value=0.80)
        config.simulation_params.category_growth = CalibratedParam(value=2.0)
        v = Validator()
        report = v.run(ValidatorInput(
            product=ProductProfile(name="x"), config=config,
        ), ctx=StageContext())
        assert any(
            "saturated markets grow slowly" in i.message
            for i in report.issues
        )

    def test_three_high_score_features_flag_anchoring(self):
        feats = [
            Feature(
                name=f"f{i}", score=0.95, polarity="positive",
                category="core_value", certainty=0.6, visibility=0.3,
                time_to_value_months=1.0,
            )
            for i in range(4)
        ]
        config = _valid_config(features=feats)
        v = Validator()
        report = v.run(ValidatorInput(
            product=ProductProfile(name="x"), config=config,
        ), ctx=StageContext())
        assert any("anchoring" in i.message for i in report.issues)


class TestPublishedBoundsHelper:
    def test_populate_published_bounds_fills_assumptions(self):
        from mindsim.models.config import Assumption
        from mindsim.pipeline.validator import populate_published_bounds

        config = _valid_config()
        config.assumptions = [
            Assumption(
                id="A1", parameter="present_bias_beta",
                value=0.55, basis="b", confidence=0.5,
            ),
            Assumption(
                id="A2", parameter="not_in_table",
                value=1.0, basis="b", confidence=0.5,
            ),
        ]
        n = populate_published_bounds(config)
        assert n == 1
        assert config.assumptions[0].published_bounds == (0.3, 0.95)
        assert config.assumptions[1].published_bounds is None


class TestConfidenceScore:
    def test_score_drops_with_errors(self):
        # Force a feature_matrix range error.
        feats = [
            Feature(
                name="f", score=2.0, polarity="positive",
                category="core_value", certainty=0.6, visibility=0.3,
                time_to_value_months=1.0,
            )
        ] * 5
        config = _valid_config(features=feats)
        v = Validator()
        report = v.run(ValidatorInput(
            product=ProductProfile(name="x"), config=config,
        ), ctx=StageContext())
        assert report.confidence_score < 1.0
