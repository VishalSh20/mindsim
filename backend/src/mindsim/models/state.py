"""Agent phase state enum (Wave 3).

Maps to AGENT_DTYPE.phase (int8) in engine/population.py. Integer values
are load-bearing — do NOT renumber without coordinating with state_machine.py
and any pickled session-state (.npz files from Wave 6).

Transitions per MECHANICS-v2.md §Phase Transitions:

    UNAWARE ──► AWARE ──► CONSIDERING ──► TRIALING ──► ADOPTED ──► LOCKED_IN
                                              │           │
                                              ▼           ▼
                                            QUIT        CHURNED

QUIT is modelled as a return to UNAWARE with trial_outcome=0; we don't
need a separate phase ID for it. CHURNED is distinct because churned
adopters retain memory (investment_depth > 0, trial_outcome may be set).
"""
from __future__ import annotations

from enum import IntEnum


class Phase(IntEnum):
    UNAWARE = 0
    AWARE = 1
    CONSIDERING = 2
    TRIALING = 3
    ADOPTED = 4
    LOCKED_IN = 5
    CHURNED = 6


# Phases where forces/decisions still apply — the decision pool.
# ADOPTED/LOCKED_IN have already decided; CHURNED opted out; UNAWARE don't know.
DECISION_PHASES: tuple[int, ...] = (
    Phase.AWARE.value,
    Phase.CONSIDERING.value,
    Phase.TRIALING.value,
)

# Phases that count as "having adopted at this round" for adoption metrics.
ADOPTED_PHASES: tuple[int, ...] = (
    Phase.ADOPTED.value,
    Phase.LOCKED_IN.value,
)

# Phases that count as "aware of the product" (for legacy `aware` boolean).
# CHURNED agents remember the product → still aware.
AWARE_PHASES: tuple[int, ...] = tuple(
    p.value for p in Phase if p != Phase.UNAWARE
)


def phase_name(p: int) -> str:
    """Human-readable phase name (for logging and RoundSnapshot keys)."""
    try:
        return Phase(p).name.lower()
    except ValueError:
        return f"unknown_{p}"
