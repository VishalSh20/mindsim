"""SimulationConfig — all numerical parameters with basis and confidence.

This is the ONLY place numerical simulation parameters live.
Each parameter has a value, basis (why this number), and confidence (how sure).
Optionally, parameters can have per-archetype overrides (by_archetype).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# The 5 Rogers adoption archetypes
ARCHETYPE_NAMES = ["innovator", "early_adopter", "early_majority", "late_majority", "laggard"]


def resolve_archetype_value(
    base_value: float,
    by_archetype: dict[str, float] | None,
    archetype_name: str,
) -> float:
    """Resolve a parameter value for a specific archetype.

    Returns the archetype-specific override if present, otherwise the base value.
    Handles None by_archetype, missing keys, and partial overrides.
    """
    if by_archetype is None:
        return base_value
    return by_archetype.get(archetype_name, base_value)


class CalibratedParam(BaseModel):
    """A single calibrated parameter with provenance."""

    value: float
    basis: str = ""  # why this number
    confidence: float = 0.5  # 0-1, how confident we are
    by_archetype: dict[str, float] | None = None  # optional per-archetype overrides


class ReferencePriceComponent(BaseModel):
    """A single component contributing to the reference price."""

    source: str  # e.g. "Free tools", "Copilot $10"
    price: float
    weight: float  # 0-1


class ReferencePriceParam(BaseModel):
    """Reference price with component breakdown."""

    value: float
    components: list[ReferencePriceComponent] = Field(default_factory=list)
    confidence: float = 0.5
    basis: str = ""
    by_archetype: dict[str, float] | None = None  # optional per-archetype overrides


class AwarenessByArchetype(BaseModel):
    """Product awareness probability by archetype."""

    innovator: float = 0.90
    early_adopter: float = 0.65
    early_majority: float = 0.30
    late_majority: float = 0.10
    laggard: float = 0.02


class PopulationConfig(BaseModel):
    """Configuration for population generation."""

    income_mean_log: float = 11.0  # log of median income
    income_sigma: float = 0.7
    market_segment: str = "general"


class Assumption(BaseModel):
    """A challengeable assumption with an ID for the user."""

    id: str  # "A1", "A2", etc.
    parameter: str
    value: float
    basis: str
    confidence: float
    sensitivity: str = "unknown"  # "high", "medium", "low"


class SimulationParams(BaseModel):
    """All numerical parameters for the simulation engine."""

    # Product economics
    price: float = 0.0
    reference_price: ReferencePriceParam = Field(
        default_factory=lambda: ReferencePriceParam(value=0.0)
    )

    # Market context
    category_penetration: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.15)
    )
    category_growth: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.5)
    )

    # Product attributes (0-1 scale)
    benefit_certainty: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.5)
    )
    perceived_benefit: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.6)
    )
    time_to_value: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.3)
    )
    requires_behavior_change: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.3)
    )
    switching_cost: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.4)
    )
    social_visibility: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.5)
    )
    identity_signal: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.3)
    )
    present_bias_beta: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.75)
    )
    fomo_intensity: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.4)
    )
    product_adoption_rate: CalibratedParam = Field(
        default_factory=lambda: CalibratedParam(value=0.05)
    )

    # Awareness
    awareness: AwarenessByArchetype = Field(default_factory=AwarenessByArchetype)

    # v2-middle Wave 1: modelling constants, not LLM-tuned.
    # Stored on the config so downstream waves can read them uniformly.
    consideration_threshold: float = 0.45  # set by maturity in Wave 5; consumed in Wave 3
    probability_weighting_gamma: float = 0.61  # Tversky & Kahneman 1992


class SimulationConfig(BaseModel):
    """Complete simulation config — output of the calibrate stage."""

    simulation_params: SimulationParams = Field(default_factory=SimulationParams)
    population_config: PopulationConfig = Field(default_factory=PopulationConfig)
    assumptions: list[Assumption] = Field(default_factory=list)
