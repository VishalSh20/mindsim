"""Tests for the per-agent feature-weight stamping (Wave 2 Commit 2).

Weights are loaded from config/feature_weights.yaml, jittered per-agent,
then renormalised so each agent's four category weights sum to 1.0.
"""
import numpy as np

from mindsim.engine.feature_weights import FEATURE_CATEGORIES, load_feature_weights
from mindsim.engine.population import AGENT_DTYPE, generate_population


class TestFeatureWeightFields:
    def test_agent_dtype_has_all_four_weight_fields(self):
        expected = {f"feature_weight_{c}" for c in FEATURE_CATEGORIES}
        assert expected.issubset(set(AGENT_DTYPE.names))

    def test_weights_sum_to_one_per_agent(self):
        rng = np.random.default_rng(42)
        agents = generate_population(n=500, rng=rng)
        totals = np.zeros(len(agents), dtype=np.float64)
        for c in FEATURE_CATEGORIES:
            totals += agents[f"feature_weight_{c}"].astype(np.float64)
        # Each agent's row sums to 1.0 within float32 tolerance.
        assert np.allclose(totals, 1.0, atol=1e-5)

    def test_weights_non_negative(self):
        rng = np.random.default_rng(7)
        agents = generate_population(n=200, rng=rng)
        for c in FEATURE_CATEGORIES:
            assert (agents[f"feature_weight_{c}"] >= 0.0).all()

    def test_innovator_core_value_higher_than_laggard(self):
        """Population-level: innovators weight core_value more than laggards."""
        rng = np.random.default_rng(0)
        agents = generate_population(n=5000, rng=rng)
        inno_mask = agents["archetype_id"] == 0  # innovator
        lag_mask = agents["archetype_id"] == 4  # laggard
        inno_mean = agents["feature_weight_core_value"][inno_mask].mean()
        lag_mean = agents["feature_weight_core_value"][lag_mask].mean()
        # Base values: innovator 0.50, laggard 0.20 — jitter won't close the gap.
        assert inno_mean > lag_mean + 0.1

    def test_laggard_ongoing_cost_higher_than_innovator(self):
        rng = np.random.default_rng(1)
        agents = generate_population(n=5000, rng=rng)
        inno_mask = agents["archetype_id"] == 0
        lag_mask = agents["archetype_id"] == 4
        inno_mean = agents["feature_weight_ongoing_cost"][inno_mask].mean()
        lag_mean = agents["feature_weight_ongoing_cost"][lag_mask].mean()
        # Base: innovator 0.10, laggard 0.55.
        assert lag_mean > inno_mean + 0.2

    def test_per_agent_jitter_produces_variation(self):
        """Agents of the same archetype should not all be carbon copies."""
        rng = np.random.default_rng(3)
        agents = generate_population(n=1000, rng=rng)
        # Look at early_majority (largest archetype)
        mask = agents["archetype_id"] == 2
        assert mask.sum() > 100
        values = agents["feature_weight_core_value"][mask]
        # Expect some std dev — noise sigma 0.05 at the source.
        assert values.std() > 0.005
        assert values.std() < 0.15  # but not wild

    def test_archetype_population_means_approximate_yaml_base(self):
        """Averaged over many agents, per-archetype mean ≈ YAML base weight."""
        fw = load_feature_weights()
        rng = np.random.default_rng(42)
        agents = generate_population(n=10_000, rng=rng)
        archetype_names = ["innovator", "early_adopter", "early_majority",
                           "late_majority", "laggard"]
        for i, aname in enumerate(archetype_names):
            mask = agents["archetype_id"] == i
            if mask.sum() < 50:
                continue
            for cat in FEATURE_CATEGORIES:
                mean = agents[f"feature_weight_{cat}"][mask].mean()
                expected = fw.by_archetype[aname][cat]
                # With jitter + renormalisation, the mean can drift slightly
                # from the pre-jitter base. Allow 0.05 slack.
                assert abs(mean - expected) < 0.05, (
                    f"{aname}.{cat}: got {mean:.3f}, expected ~{expected:.3f}"
                )
