"""SimulationResult + AnalysisReport — output data structures."""

from pydantic import BaseModel, Field


class ForceDecomposition(BaseModel):
    """Weighted force averages for display."""

    prospect_value: float = 0.0
    anchoring: float = 0.0
    status_quo: float = 0.0
    social_proof: float = 0.0
    fomo: float = 0.0
    hyperbolic_discounting: float = 0.0
    identity_signaling: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "prospect_value": self.prospect_value,
            "anchoring": self.anchoring,
            "status_quo": self.status_quo,
            "social_proof": self.social_proof,
            "fomo": self.fomo,
            "hyperbolic_discounting": self.hyperbolic_discounting,
            "identity_signaling": self.identity_signaling,
        }


class ArchetypeSegment(BaseModel):
    """Adoption rate for a single archetype."""

    name: str
    adoption_rate: float
    count: int
    total: int


class IncomeSegment(BaseModel):
    """Adoption rate for an income bracket."""

    bracket: str  # "high", "mid", "low"
    adoption_rate: float
    count: int
    total: int


class ProximitySegment(BaseModel):
    """Agents grouped by decision proximity."""

    locked: float = 0.0       # prob > 0.7
    convertible: float = 0.0  # prob 0.3 - 0.7
    unreachable: float = 0.0  # prob < 0.3


class EventForceAdjustment(BaseModel):
    """Per-force adjustment from an event."""

    force: str
    magnitude: float
    mechanism: str
    new_value: float | None = None  # for reference price changes


class EventResult(BaseModel):
    """Before/after comparison from an event."""

    event_text: str
    adoption_before: float
    adoption_after: float
    adoption_delta: float
    force_adjustments: list[EventForceAdjustment] = Field(default_factory=list)
    segment_effects: dict[str, float] = Field(default_factory=dict)
    second_order_effects: list[str] = Field(default_factory=list)


class SimulationResult(BaseModel):
    """Output of the simulate stage."""

    # Headline numbers
    total_adoption: float = 0.0      # of all agents
    aware_adoption: float = 0.0      # of aware agents only
    n_agents: int = 1000
    n_aware: int = 0

    # Force decomposition (boundary-weighted)
    force_decomposition: ForceDecomposition = Field(
        default_factory=ForceDecomposition
    )

    # Segmentation
    by_archetype: list[ArchetypeSegment] = Field(default_factory=list)
    by_income: list[IncomeSegment] = Field(default_factory=list)
    by_proximity: ProximitySegment = Field(default_factory=ProximitySegment)

    # Event results
    event_results: list[EventResult] = Field(default_factory=list)

    # Raw data for deep dives (not serialized by default)
    _agent_forces: dict | None = None
    _agent_probs: object = None
    _agent_decisions: object = None
    _agents: object = None


class SensitivityResult(BaseModel):
    """Sensitivity analysis for a single parameter."""

    parameter: str
    base_adoption: float
    low_adoption: float   # param -30%
    high_adoption: float  # param +30%
    swing: float          # high - low in percentage points
    confidence: float


class ConfidenceBand(BaseModel):
    """Overall confidence band for the adoption estimate."""

    central: float
    low: float
    high: float
    drivers: list[str] = Field(default_factory=list)  # params driving uncertainty


class InterventionResult(BaseModel):
    """Ranked intervention with predicted lift."""

    name: str
    description: str
    adoption_before: float
    adoption_after: float
    lift_pp: float  # percentage points
    mechanism: str


class AnalysisReport(BaseModel):
    """Final output — everything the user sees."""

    # Behavioral audit prose
    text: str = ""

    # Quantitative analysis
    sensitivity: list[SensitivityResult] = Field(default_factory=list)
    confidence_band: ConfidenceBand = Field(
        default_factory=lambda: ConfidenceBand(central=0.0, low=0.0, high=0.0)
    )
    interventions: list[InterventionResult] = Field(default_factory=list)

    # Assumptions table
    assumptions: list[dict] = Field(default_factory=list)
