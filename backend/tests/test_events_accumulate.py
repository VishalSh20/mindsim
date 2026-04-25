"""Wave 6 — events mutate persisted population in place.

Verifies the headline Wave 6 properties:

  * `process_event` does NOT regenerate agents — adopted/locked-in
    agents from the prior simulation persist across the event call.
  * Event order matters: applying event A then B produces a different
    final adoption rate than B then A on the same starting state, when
    the events have asymmetric effects (the canonical "launch + bad
    press" check called out in WAVES-4-TO-8.md exit criteria).
  * Per-event LLM is called once and receives the trajectory snapshot
    (phase distribution) in its user message.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from mindsim.models.config import (
    CalibratedParam,
    PopulationConfig,
    ReferencePriceParam,
    SimulationConfig,
    SimulationParams,
)
from mindsim.models.product import Feature
from mindsim.models.state import Phase
from mindsim.pipeline.events import process_event
from mindsim.pipeline.simulate import simulate


class _ScriptedLLM:
    """Stub LLM that returns a fixed payload (or one chosen by event_text)."""

    def __init__(
        self,
        payload: dict | None = None,
        chooser: Callable[[str], dict] | None = None,
    ):
        self.payload = payload or {}
        self.chooser = chooser
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system_prompt: str, user_message: str, temperature: float = 0.2):
        # Recover the event text from the front of the user_message so the
        # chooser can return event-specific responses.
        event_line = ""
        for line in user_message.splitlines():
            if line.startswith('"') and line.endswith('"'):
                event_line = line.strip('"')
                break
        self.calls.append((event_line, user_message))
        if self.chooser is not None:
            return self.chooser(event_line)
        return self.payload


def _config() -> SimulationConfig:
    return SimulationConfig(
        simulation_params=SimulationParams(
            price=20.0,
            reference_price=ReferencePriceParam(value=18.0, basis="ref", confidence=0.6),
            category_penetration=CalibratedParam(value=0.35, confidence=0.6),
            category_growth=CalibratedParam(value=0.4, confidence=0.5),
            perceived_benefit=CalibratedParam(value=0.7, confidence=0.6),
            benefit_certainty=CalibratedParam(value=0.6, confidence=0.6),
            switching_cost=CalibratedParam(value=0.4, confidence=0.6),
            social_visibility=CalibratedParam(value=0.5, confidence=0.6),
            identity_signal=CalibratedParam(value=0.3, confidence=0.6),
            present_bias_beta=CalibratedParam(value=0.75, confidence=0.6),
            fomo_intensity=CalibratedParam(value=0.4, confidence=0.6),
            product_adoption_rate=CalibratedParam(value=0.05, confidence=0.5),
            requires_behavior_change=CalibratedParam(value=0.3, confidence=0.5),
            # Leave feature_matrix empty so the engine uses the scalar
            # `perceived_benefit` path; events in this test exercise the
            # scalar restamping flow.
            feature_matrix=[],
        ),
        population_config=PopulationConfig(),
    )


def _seed_run(rng_seed: int = 42, n_agents: int = 400, n_rounds: int = 8):
    rng = np.random.default_rng(rng_seed)
    config = _config()
    result = simulate(config, n_agents=n_agents, rng=rng, n_rounds=n_rounds)
    return result, config, rng


class TestPopulationPersists:
    def test_adopted_and_locked_agents_carry_over(self):
        result, config, rng = _seed_run()
        agents = result._agents
        assert agents is not None

        # Snapshot which agents finished as ADOPTED or LOCKED_IN.
        decided_mask_before = np.isin(
            agents["phase"],
            np.array([int(Phase.ADOPTED), int(Phase.LOCKED_IN)], dtype=agents["phase"].dtype),
        )
        decided_idx_before = set(np.flatnonzero(decided_mask_before).tolist())
        assert len(decided_idx_before) > 0, "seed run produced no adopters"

        # Apply an event with a stub LLM that produces no param changes.
        llm = _ScriptedLLM(payload={
            "force_adjustments": [],
            "segment_effects": {},
            "second_order_effects": [],
        })
        new_result, _ = process_event("nothing happens", result, config, llm, rng)

        # Same backing array (mutate-in-place contract).
        assert new_result._agents is agents

        # Decided agents stay decided modulo a small per-round churn fraction.
        decided_mask_after = np.isin(
            agents["phase"],
            np.array(
                [int(Phase.ADOPTED), int(Phase.LOCKED_IN)],
                dtype=agents["phase"].dtype,
            ),
        )
        decided_idx_after = set(np.flatnonzero(decided_mask_after).tolist())
        retained = decided_idx_before & decided_idx_after
        # Allow up to ~10% churn over 2 sub-rounds at base_churn=0.02.
        assert len(retained) >= 0.85 * len(decided_idx_before)

    def test_event_history_appends_not_overwrites(self):
        result, config, rng = _seed_run()
        llm = _ScriptedLLM(payload={"force_adjustments": []})
        r1, e1 = process_event("first", result, config, llm, rng)
        r2, e2 = process_event("second", r1, config, llm, rng)
        assert [er.event_text for er in r2.event_results] == ["first", "second"]


class TestEventOrderMatters:
    """Apply two asymmetric events in opposing orders; results should diverge."""

    def _scripted_llm(self) -> _ScriptedLLM:
        def chooser(event_text: str) -> dict:
            if "launches" in event_text.lower():
                return {
                    "force_adjustments": [
                        {
                            "force": "prospect_value",
                            "param_changes": {"perceived_benefit": 0.85},
                            "magnitude": 0.20,
                            "mechanism": "launch lifts perceived benefit",
                        }
                    ],
                    "segment_effects": {},
                    "second_order_effects": [],
                }
            if "bad press" in event_text.lower():
                return {
                    "force_adjustments": [
                        {
                            "force": "prospect_value",
                            "param_changes": {"perceived_benefit": 0.40},
                            "magnitude": -0.25,
                            "mechanism": "press cuts perceived benefit",
                        }
                    ],
                    "segment_effects": {},
                    "second_order_effects": [],
                }
            return {"force_adjustments": []}
        return _ScriptedLLM(chooser=chooser)

    def test_launch_then_press_differs_from_press_then_launch(self):
        # Seed with a SHORT run so there's still a large convertible pool
        # the events can move; with the default 8 rounds nearly every
        # decideable agent has already settled and the events would only
        # nibble at the churn/recovery margins.
        result_a, config_a, rng_a = _seed_run(rng_seed=99, n_agents=600, n_rounds=2)
        result_b, config_b, rng_b = _seed_run(rng_seed=99, n_agents=600, n_rounds=2)

        # Sanity: identical baselines.
        assert result_a.total_adoption == pytest.approx(result_b.total_adoption)

        llm_a = self._scripted_llm()
        llm_b = self._scripted_llm()

        # 6 sub-rounds per event so launch-driven adopters have time to
        # reach LOCK_IN_THRESHOLD (investment_depth grows 0.1/round from a
        # 0.1 seed → ~5-6 rounds to lock in) before the second event hits.
        n_sub = 6

        # Order 1: launch → bad press
        r1, _ = process_event("Product A launches", result_a, config_a, llm_a, rng_a, n_subrounds=n_sub)
        r1_final, _ = process_event(
            "Product A gets bad press", r1, config_a, llm_a, rng_a, n_subrounds=n_sub
        )

        # Order 2: bad press → launch (reversed)
        r2, _ = process_event(
            "Product A gets bad press", result_b, config_b, llm_b, rng_b, n_subrounds=n_sub
        )
        r2_final, _ = process_event(
            "Product A launches", r2, config_b, llm_b, rng_b, n_subrounds=n_sub
        )

        # The headline Wave 6 invariant: event ORDER MATTERS — the same two
        # events applied in different orders must produce a measurably
        # different end-state. Direction of the gap is a property of the
        # particular events: launch→press locks adopters in before the
        # press hits and they can only churn at base_churn=0.02/round, so
        # this order should retain more adopters than the reverse.
        assert r1_final.total_adoption != pytest.approx(r2_final.total_adoption, abs=0.005)
        assert r1_final.total_adoption > r2_final.total_adoption


class TestPromptCarriesTrajectory:
    def test_user_message_includes_phase_breakdown(self):
        result, config, rng = _seed_run()
        llm = _ScriptedLLM(payload={"force_adjustments": []})
        process_event("inspect the prompt", result, config, llm, rng)
        assert len(llm.calls) == 1
        _, user_msg = llm.calls[0]
        assert "Phase Distribution" in user_msg
        assert "Per-Archetype Phase Breakdown" in user_msg
