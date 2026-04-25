"""Wave 8 — per-archetype segment narrative tests."""
from __future__ import annotations

import numpy as np
import pytest

from mindsim.engine.archetypes import load_archetypes
from mindsim.models.config import (
    CalibratedParam,
    PopulationConfig,
    ReferencePriceParam,
    SimulationConfig,
    SimulationParams,
)
from mindsim.models.results import SimulationResult
from mindsim.pipeline.segments import build_segment_report
from mindsim.pipeline.simulate import simulate


def _config():
    return SimulationConfig(
        simulation_params=SimulationParams(
            price=20.0,
            reference_price=ReferencePriceParam(value=18.0, basis="r"),
            category_penetration=CalibratedParam(value=0.30),
            category_growth=CalibratedParam(value=0.5),
            perceived_benefit=CalibratedParam(value=0.7),
            benefit_certainty=CalibratedParam(value=0.6),
            switching_cost=CalibratedParam(value=0.4),
            social_visibility=CalibratedParam(value=0.5),
            identity_signal=CalibratedParam(value=0.3),
            present_bias_beta=CalibratedParam(value=0.75),
            fomo_intensity=CalibratedParam(value=0.4),
            product_adoption_rate=CalibratedParam(value=0.05),
            requires_behavior_change=CalibratedParam(value=0.3),
            feature_matrix=[],
        ),
        population_config=PopulationConfig(),
    )


class TestSegmentReport:
    def test_returns_one_narrative_per_archetype(self):
        rng = np.random.default_rng(2)
        result = simulate(_config(), n_agents=400, rng=rng, n_rounds=4)
        report = build_segment_report(result)
        names = {s.archetype for s in report.segments}
        # All 5 Rogers archetypes should be represented when n=400.
        assert names == set(load_archetypes().names)

    def test_some_narratives_carry_dominant_forces(self):
        # Force values come from compute_forces, which only populates the
        # decision pool (AWARE / CONSIDERING / TRIALING). Innovators that
        # have already adopted by the final round have all-NaN forces,
        # so their dominant_driver/blocker stay "". At least the
        # archetypes still in the decision pool must carry forces.
        rng = np.random.default_rng(3)
        result = simulate(_config(), n_agents=400, rng=rng, n_rounds=4)
        report = build_segment_report(result)
        with_forces = [
            s for s in report.segments
            if s.dominant_driver and s.dominant_blocker
        ]
        assert with_forces, "expected at least one archetype with force data"
        for seg in with_forces:
            # Driver mean ≥ blocker mean by construction.
            assert seg.dominant_driver_value >= seg.dominant_blocker_value

    def test_trajectories_populated(self):
        rng = np.random.default_rng(4)
        result = simulate(_config(), n_agents=400, rng=rng, n_rounds=4)
        report = build_segment_report(result)
        for seg in report.segments:
            assert 1 <= len(seg.representative_trajectories) <= 3
            for t in seg.representative_trajectories:
                # Each trajectory line names a phase ("aware", "considering", etc.)
                assert any(
                    p in t for p in (
                        "unaware", "aware", "considering", "trialing",
                        "adopted", "locked_in", "churned",
                    )
                )

    def test_empty_when_agents_missing(self):
        result = SimulationResult(total_adoption=0.0)
        report = build_segment_report(result)
        assert report.segments == []

    def test_adoption_rates_match_simulation(self):
        rng = np.random.default_rng(5)
        result = simulate(_config(), n_agents=400, rng=rng, n_rounds=4)
        report = build_segment_report(result)
        # Reconstruct total adoption as a weighted average across narratives.
        total_count = sum(s.count for s in report.segments)
        total_total = sum(s.total for s in report.segments)
        recovered = total_count / max(total_total, 1)
        assert recovered == pytest.approx(result.total_adoption, abs=1e-6)
