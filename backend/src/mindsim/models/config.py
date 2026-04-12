"""SimulationConfig — all numerical parameters with basis and confidence.

This is the ONLY place numerical simulation parameters live.
Each parameter has a value, basis (why this number), and confidence (how sure).
"""

from pydantic import BaseModel, Field


class CalibratedParam(BaseModel):
    """A single calibrated parameter with provenance."""

    value: float
    basis: str = ""  # why this number
    confidence: float = 0.5  # 0-1, how confident we are


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


class SimulationConfig(BaseModel):
    """Complete simulation config — output of the calibrate stage."""

    simulation_params: SimulationParams = Field(default_factory=SimulationParams)
    population_config: PopulationConfig = Field(default_factory=PopulationConfig)
    assumptions: list[Assumption] = Field(default_factory=list)
