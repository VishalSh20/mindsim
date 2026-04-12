"""Test pipeline — end-to-end with mocked LLM."""

import json

import numpy as np
import pytest

from mindsim.models.config import SimulationConfig
from mindsim.models.market import MarketContext
from mindsim.models.product import ProductProfile
from mindsim.pipeline.simulate import simulate


class TestSimulatePipeline:
    """Test the simulate stage with pre-built config."""

    def test_simulate_produces_result(self):
        """Simulate with default config produces valid SimulationResult."""
        config = SimulationConfig()
        config.simulation_params.price = 20.0
        config.simulation_params.reference_price.value = 15.0
        config.simulation_params.perceived_benefit.value = 0.6

        result = simulate(config, n_agents=500, rng=np.random.default_rng(42))

        assert 0 <= result.total_adoption <= 1
        assert 0 <= result.aware_adoption <= 1
        assert result.n_agents == 500
        assert result.n_aware > 0
        assert len(result.by_archetype) == 5
        assert len(result.by_income) == 3

    def test_proximity_sums_to_one(self):
        """Locked + convertible + unreachable should sum to ~1."""
        config = SimulationConfig()
        config.simulation_params.price = 20.0

        result = simulate(config, n_agents=1000, rng=np.random.default_rng(42))
        prox = result.by_proximity
        total = prox.locked + prox.convertible + prox.unreachable
        assert abs(total - 1.0) < 0.02, f"Proximity sum = {total}, expected ~1.0"

    def test_skip_research_config_works(self):
        """SimulationConfig with all defaults should produce valid results."""
        config = SimulationConfig()
        config.simulation_params.price = 10.0
        config.simulation_params.perceived_benefit.value = 0.7
        config.simulation_params.reference_price.value = 20.0

        result = simulate(config, rng=np.random.default_rng(42))
        assert result.total_adoption > 0
        assert result.force_decomposition is not None
