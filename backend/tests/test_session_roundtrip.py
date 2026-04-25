"""Wave 6 — session save/load round-trip tests.

Verifies:
  * agents survive .npz save → load byte-for-byte (every field).
  * SimulationConfig round-trips through Pydantic JSON.
  * meta envelope (run_id, parent_run_id, event_history) is preserved.
  * mismatched SCHEMA_VERSION refuses to load with SessionVersionError.
  * mismatched dtype descriptor refuses to load.
  * missing file raises FileNotFoundError.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mindsim.engine.population import AGENT_DTYPE, generate_population
from mindsim.models.config import (
    CalibratedParam,
    PopulationConfig,
    ReferencePriceParam,
    SimulationConfig,
    SimulationParams,
)
from mindsim.pipeline import session as session_mod
from mindsim.pipeline.session import (
    SessionVersionError,
    append_event_to_history,
    load_session,
    new_run_id,
    save_session,
)


def _build_config() -> SimulationConfig:
    params = SimulationParams(
        price=20.0,
        reference_price=ReferencePriceParam(value=15.0, basis="ref", confidence=0.7),
        category_penetration=CalibratedParam(value=0.4, basis="b", confidence=0.6),
    )
    return SimulationConfig(simulation_params=params, population_config=PopulationConfig())


def _agents(n: int = 64) -> np.ndarray:
    rng = np.random.default_rng(123)
    return generate_population(
        n=n,
        population_config=PopulationConfig(),
        sim_params=_build_config().simulation_params,
        rng=rng,
    )


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Field-by-field equality on a structured array."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    for name in a.dtype.names:
        if not np.array_equal(a[name], b[name], equal_nan=True):
            return False
    return True


class TestRoundTrip:
    def test_save_and_load_preserves_agents(self, tmp_path: Path):
        agents = _agents(50)
        config = _build_config()
        run_id = "run-test-aaaa"
        path = save_session(tmp_path / "s.npz", agents, config, run_id=run_id)

        state = load_session(path)
        assert state.run_id == run_id
        assert state.parent_run_id is None
        assert state.n_agents == 50
        assert _arrays_equal(state.agents, agents)

    def test_config_round_trips_through_json(self, tmp_path: Path):
        agents = _agents(20)
        config = _build_config()
        path = save_session(tmp_path / "s.npz", agents, config, run_id="r1")
        state = load_session(path)
        # Pydantic equality is deep on field-by-field values.
        assert state.config.simulation_params.price == config.simulation_params.price
        assert (
            state.config.simulation_params.reference_price.basis
            == config.simulation_params.reference_price.basis
        )

    def test_event_history_preserved(self, tmp_path: Path):
        agents = _agents(20)
        history = append_event_to_history([], "price drop", 0.10, 0.18)
        history = append_event_to_history(history, "bad press", 0.18, 0.13)
        path = save_session(
            tmp_path / "s.npz",
            agents,
            _build_config(),
            run_id="r1",
            parent_run_id="r0",
            event_history=history,
        )
        state = load_session(path)
        assert state.parent_run_id == "r0"
        assert len(state.event_history) == 2
        assert state.event_history[0]["event_text"] == "price drop"
        assert state.event_history[1]["adoption_after"] == pytest.approx(0.13)

    def test_npz_suffix_added_when_missing(self, tmp_path: Path):
        agents = _agents(10)
        path = save_session(tmp_path / "noext", agents, _build_config(), run_id="r")
        assert path.suffix == ".npz"
        assert path.exists()


class TestSchemaGuards:
    def test_schema_version_mismatch_refuses(self, tmp_path: Path, monkeypatch):
        agents = _agents(10)
        path = save_session(tmp_path / "s.npz", agents, _build_config(), run_id="r")

        # Bump the version after saving so the load sees a stale file.
        monkeypatch.setattr(session_mod, "SCHEMA_VERSION", session_mod.SCHEMA_VERSION + 1)
        with pytest.raises(SessionVersionError, match="schema_version"):
            load_session(path)

    def test_dtype_descriptor_mismatch_refuses(self, tmp_path: Path, monkeypatch):
        agents = _agents(10)
        path = save_session(tmp_path / "s.npz", agents, _build_config(), run_id="r")

        # Pretend AGENT_DTYPE changed by patching the descriptor helper to
        # return something different on load. Easier: rewrite the meta JSON
        # in-place with a corrupted descriptor.
        with np.load(path, allow_pickle=True) as data:
            agents_arr = data["agents"]
            meta = json.loads(str(data["meta"].item()))
        meta["agent_dtype_descr"][0] = ["bogus_field", "<i8"]
        np.savez_compressed(
            path, agents=agents_arr, meta=np.array(json.dumps(meta), dtype=object)
        )
        with pytest.raises(SessionVersionError, match="agent_dtype_descr"):
            load_session(path)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_session(tmp_path / "nope.npz")

    def test_dtype_check_blocks_save_of_wrong_array(self, tmp_path: Path):
        wrong = np.zeros(5, dtype=[("a", np.int32)])
        with pytest.raises(ValueError, match="dtype mismatch"):
            save_session(tmp_path / "s.npz", wrong, _build_config(), run_id="r")


class TestRunIdGeneration:
    def test_new_run_id_is_unique(self):
        a = new_run_id()
        b = new_run_id()
        assert a != b
        assert a.startswith("run-")
