"""Tests for the Feature schema and the feature_weights YAML loader."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from mindsim.engine.feature_weights import (
    DEFAULT_PATH,
    FEATURE_CATEGORIES,
    FeatureWeights,
    jitter_and_renormalise,
    load_feature_weights,
)
from mindsim.models.config import ARCHETYPE_NAMES, SimulationParams
from mindsim.models.product import CompetitorInfo, Feature


class TestFeatureSchema:
    def test_minimal_feature(self):
        f = Feature(name="core", score=0.8, polarity="positive", category="core_value")
        assert f.certainty == 0.5
        assert f.evidence_refs == []

    def test_polarity_enum(self):
        with pytest.raises(ValidationError):
            Feature(name="x", score=0.5, polarity="maybe", category="core_value")

    def test_category_enum(self):
        with pytest.raises(ValidationError):
            Feature(name="x", score=0.5, polarity="positive", category="bogus")

    def test_competitor_feature_scores_defaults_empty(self):
        c = CompetitorInfo(name="VS Code")
        assert c.feature_scores == {}

    def test_competitor_feature_scores_populated(self):
        c = CompetitorInfo(
            name="Cursor", feature_scores={"code_quality": 0.75, "price": 0.4}
        )
        assert c.feature_scores["code_quality"] == 0.75

    def test_simulation_params_feature_matrix_defaults_empty(self):
        p = SimulationParams()
        assert p.feature_matrix == []

    def test_simulation_params_accepts_features(self):
        p = SimulationParams(
            feature_matrix=[
                Feature(name="a", score=0.6, polarity="positive", category="core_value"),
                Feature(
                    name="b", score=0.3, polarity="negative", category="ongoing_cost"
                ),
            ]
        )
        assert len(p.feature_matrix) == 2


class TestFeatureWeightsYAML:
    def test_default_yaml_loads(self):
        fw = load_feature_weights()
        assert isinstance(fw, FeatureWeights)

    def test_all_archetypes_present(self):
        fw = load_feature_weights()
        for a in ARCHETYPE_NAMES:
            assert a in fw.by_archetype

    def test_rows_sum_to_one(self):
        fw = load_feature_weights()
        for a in ARCHETYPE_NAMES:
            total = sum(fw.by_archetype[a].values())
            assert abs(total - 1.0) < 1e-6, f"{a} sums to {total}"

    def test_categories_match_contract(self):
        fw = load_feature_weights()
        for a in ARCHETYPE_NAMES:
            assert set(fw.by_archetype[a].keys()) == set(FEATURE_CATEGORIES)

    def test_archetype_ordering_matches_intuition(self):
        """Innovators weight core_value higher than laggards do."""
        fw = load_feature_weights()
        assert (
            fw.by_archetype["innovator"]["core_value"]
            > fw.by_archetype["laggard"]["core_value"]
        )

    def test_laggards_weight_cost_highest(self):
        """Laggards should weight ongoing_cost higher than innovators."""
        fw = load_feature_weights()
        assert (
            fw.by_archetype["laggard"]["ongoing_cost"]
            > fw.by_archetype["innovator"]["ongoing_cost"]
        )

    def test_rejects_bad_sum(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        payload = {
            "archetypes": {
                a: {c: 0.5 for c in FEATURE_CATEGORIES} for a in ARCHETYPE_NAMES
            }
        }
        bad.write_text(yaml.safe_dump(payload))
        with pytest.raises(ValueError, match="sum"):
            load_feature_weights(bad)

    def test_rejects_missing_archetype(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        payload = {
            "archetypes": {
                "innovator": {c: 0.25 for c in FEATURE_CATEGORIES},
                # missing everyone else
            }
        }
        bad.write_text(yaml.safe_dump(payload))
        with pytest.raises(ValueError, match="missing"):
            load_feature_weights(bad)

    def test_rejects_unknown_category(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        payload = {
            "archetypes": {
                a: {
                    "core_value": 0.25,
                    "social_signal": 0.25,
                    "ongoing_cost": 0.25,
                    "switching_friction_reducer": 0.24,
                    "wat": 0.01,
                }
                for a in ARCHETYPE_NAMES
            }
        }
        bad.write_text(yaml.safe_dump(payload))
        with pytest.raises(ValueError, match="unknown"):
            load_feature_weights(bad)


class TestJitterAndRenormalise:
    def test_preserves_unit_sum(self):
        rng = np.random.default_rng(42)
        base = {c: 0.25 for c in FEATURE_CATEGORIES}
        for _ in range(10):
            out = jitter_and_renormalise(base, rng, sigma=0.1)
            assert abs(sum(out.values()) - 1.0) < 1e-9

    def test_zero_sigma_is_identity(self):
        rng = np.random.default_rng(42)
        base = {"core_value": 0.5, "social_signal": 0.2, "ongoing_cost": 0.2, "switching_friction_reducer": 0.1}
        out = jitter_and_renormalise(base, rng, sigma=0.0)
        for k, v in base.items():
            assert abs(out[k] - v) < 1e-9

    def test_keys_preserved(self):
        rng = np.random.default_rng(0)
        base = {c: 0.25 for c in FEATURE_CATEGORIES}
        out = jitter_and_renormalise(base, rng, sigma=0.05)
        assert set(out.keys()) == set(base.keys())
