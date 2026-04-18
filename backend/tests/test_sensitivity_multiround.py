"""Tests for the Wave 3 multi-round sensitivity analysis."""
import numpy as np

from mindsim.engine.sensitivity import run_sensitivity_analysis
from mindsim.models.config import (
    CalibratedParam,
    ReferencePriceParam,
    SimulationConfig,
)
from mindsim.models.product import Feature
from mindsim.pipeline.simulate import simulate


def _features():
    return [
        Feature(name="core", score=0.7, polarity="positive",
                category="core_value", certainty=0.6, visibility=0.3,
                time_to_value_months=0.5),
        Feature(name="cost", score=0.4, polarity="negative",
                category="ongoing_cost", certainty=0.7, visibility=0.1,
                time_to_value_months=0.0),
        Feature(name="integ", score=0.5, polarity="positive",
                category="switching_friction_reducer", certainty=0.6,
                visibility=0.2, time_to_value_months=0.0),
        Feature(name="sig", score=0.3, polarity="positive",
                category="social_signal", certainty=0.5, visibility=0.5,
                time_to_value_months=1.0),
    ]


def _config():
    config = SimulationConfig()
    p = config.simulation_params
    p.n_rounds = 4  # faster tests
    p.feature_matrix = _features()
    p.category_penetration = CalibratedParam(value=0.2, confidence=0.4)
    p.category_growth = CalibratedParam(value=0.5, confidence=0.4)
    p.benefit_certainty = CalibratedParam(value=0.5, confidence=0.5)
    p.perceived_benefit = CalibratedParam(value=0.6, confidence=0.5)
    p.present_bias_beta = CalibratedParam(value=0.75, confidence=0.6)
    p.reference_price = ReferencePriceParam(value=25.0, confidence=0.3)
    p.price = 20.0
    return config


def test_sensitivity_runs_multiround_and_returns_results():
    config = _config()
    rng = np.random.default_rng(42)
    result = simulate(config=config, n_agents=300, rng=rng)

    rng = np.random.default_rng(7)
    sensitivities = run_sensitivity_analysis(
        agents=result._agents,
        config=config,
        base_adoption=result.total_adoption,
        rng=rng,
        n_agents=300,
    )
    assert len(sensitivities) > 0
    for s in sensitivities:
        assert 0.0 <= s.low_adoption <= 1.0
        assert 0.0 <= s.high_adoption <= 1.0
        assert s.swing >= 0.0


def test_sensitivity_high_confidence_params_skipped():
    config = _config()
    # Bump one param to high confidence — should not appear.
    config.simulation_params.perceived_benefit = CalibratedParam(
        value=0.6, confidence=0.95
    )
    rng = np.random.default_rng(1)
    result = simulate(config=config, n_agents=200, rng=rng)
    sensitivities = run_sensitivity_analysis(
        agents=result._agents,
        config=config,
        base_adoption=result.total_adoption,
        n_agents=200,
    )
    names = {s.parameter for s in sensitivities}
    assert "perceived_benefit" not in names


def test_sensitivity_results_sorted_by_swing():
    config = _config()
    rng = np.random.default_rng(5)
    result = simulate(config=config, n_agents=300, rng=rng)
    sensitivities = run_sensitivity_analysis(
        agents=result._agents,
        config=config,
        base_adoption=result.total_adoption,
        n_agents=300,
    )
    swings = [s.swing for s in sensitivities]
    assert swings == sorted(swings, reverse=True)


def test_reproducible_with_fixed_seed():
    """Same config + same seed → same final adoption."""
    config = _config()
    a = simulate(config=config, n_agents=500, rng=np.random.default_rng(123))
    b = simulate(config=config, n_agents=500, rng=np.random.default_rng(123))
    assert abs(a.total_adoption - b.total_adoption) < 1e-9
    assert len(a.rounds) == len(b.rounds)
    for sa, sb in zip(a.rounds, b.rounds):
        assert sa.total_adoption == sb.total_adoption
