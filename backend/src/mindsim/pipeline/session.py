"""Wave 6 — session-state persistence.

A session captures everything needed to resume a simulation across CLI
invocations:
  - the agent NumPy array (phases, traits, per-agent product params)
  - the SimulationConfig that produced it (so events stay coherent)
  - run identity (run_id, parent_run_id, event history)

Format: a single `.npz` archive containing two arrays:
  - `agents`  : structured AGENT_DTYPE array
  - `meta`    : 0-d unicode array holding a JSON document with the
                config dump, manifest, schema version, and dtype
                descriptor.

Schema version is bumped whenever AGENT_DTYPE changes. Mismatched
versions refuse to load (NumPy structured-array offsets shift on dtype
edits — silent loading would corrupt agent state).
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mindsim.engine.population import AGENT_DTYPE
from mindsim.models.config import SimulationConfig

logger = logging.getLogger(__name__)


# Bump whenever AGENT_DTYPE changes shape or field types.
SCHEMA_VERSION = 1


class SessionVersionError(RuntimeError):
    """Raised when a session file's schema doesn't match the current code."""


@dataclass
class SessionState:
    """In-memory representation of a loaded session."""

    agents: np.ndarray
    config: SimulationConfig
    run_id: str
    parent_run_id: str | None = None
    created_at: str = ""
    event_history: list[dict[str, Any]] = field(default_factory=list)
    n_agents: int = 0
    schema_version: int = SCHEMA_VERSION


def new_run_id() -> str:
    """Short, sortable-by-time identifier for a fresh run."""
    return f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _agent_dtype_descr() -> list:
    """JSON-safe descriptor for the current AGENT_DTYPE."""
    return [list(field) for field in AGENT_DTYPE.descr]


def save_session(
    path: str | Path,
    agents: np.ndarray,
    config: SimulationConfig,
    run_id: str,
    parent_run_id: str | None = None,
    event_history: list[dict[str, Any]] | None = None,
) -> Path:
    """Serialise an agent array + config to a `.npz` session file.

    Args:
        path: target file path. Suffix `.npz` is added if missing.
        agents: AGENT_DTYPE structured array.
        config: SimulationConfig that produced these agents.
        run_id: identifier for this run.
        parent_run_id: id of the run we resumed from (None for first run).
        event_history: prior events applied in this session lineage.

    Returns:
        The path the file was written to.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    if agents.dtype != AGENT_DTYPE:
        raise ValueError(
            f"agent dtype mismatch: got {agents.dtype}, expected AGENT_DTYPE"
        )

    meta = {
        "schema_version": SCHEMA_VERSION,
        "agent_dtype_descr": _agent_dtype_descr(),
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_agents": int(len(agents)),
        "event_history": event_history or [],
        "config": config.model_dump(mode="json"),
    }
    meta_arr = np.array(json.dumps(meta), dtype=object)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, agents=agents, meta=meta_arr)
    logger.info("session saved: %s (run_id=%s, n=%d)", path, run_id, len(agents))
    return path


def load_session(path: str | Path) -> SessionState:
    """Load a session file written by `save_session`.

    Raises:
        FileNotFoundError: if path does not exist.
        SessionVersionError: if schema_version or AGENT_DTYPE mismatch.
        ValueError: if the file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"session file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        if "agents" not in data.files or "meta" not in data.files:
            raise ValueError(
                f"session file missing required keys (got {data.files})"
            )
        agents = data["agents"]
        meta_raw = data["meta"]

    try:
        meta_str = meta_raw.item() if meta_raw.shape == () else meta_raw[0]
        meta = json.loads(meta_str)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"session meta could not be parsed: {exc}") from exc

    schema_version = int(meta.get("schema_version", -1))
    if schema_version != SCHEMA_VERSION:
        raise SessionVersionError(
            f"session schema_version={schema_version} but code expects "
            f"{SCHEMA_VERSION}. Re-run from scratch."
        )

    saved_descr = meta.get("agent_dtype_descr", [])
    current_descr = _agent_dtype_descr()
    if saved_descr != current_descr:
        raise SessionVersionError(
            "session agent_dtype_descr does not match current AGENT_DTYPE. "
            "Bump SCHEMA_VERSION when changing the dtype."
        )

    if agents.dtype != AGENT_DTYPE:
        raise SessionVersionError(
            f"loaded agents dtype {agents.dtype} != AGENT_DTYPE"
        )

    config = SimulationConfig.model_validate(meta["config"])

    return SessionState(
        agents=agents.copy(),  # detach from the npz mmap so callers can mutate
        config=config,
        run_id=meta["run_id"],
        parent_run_id=meta.get("parent_run_id"),
        created_at=meta.get("created_at", ""),
        event_history=list(meta.get("event_history") or []),
        n_agents=int(meta.get("n_agents", len(agents))),
        schema_version=schema_version,
    )


def append_event_to_history(
    history: list[dict[str, Any]],
    event_text: str,
    adoption_before: float,
    adoption_after: float,
) -> list[dict[str, Any]]:
    """Append a serialisable event entry to a history list."""
    history = list(history)
    history.append(
        {
            "event_text": event_text,
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "adoption_before": float(adoption_before),
            "adoption_after": float(adoption_after),
        }
    )
    return history
