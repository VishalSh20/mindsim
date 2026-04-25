"""SimulationResult + AnalysisReport — output data structures."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from mindsim.models.evidence import EvidenceStore


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

    # v2-middle Wave 3: per-round snapshots from the multi-round loop.
    # Empty when simulate() ran with n_rounds=1 (legacy single-shot path).
    rounds: list["RoundSnapshot"] = Field(default_factory=list)

    # Raw data for deep dives (not serialized by default)
    _agent_forces: dict | None = None
    _agent_probs: object = None
    _agent_decisions: object = None
    _agents: object = None

    # Wave 8.5 — populated by analyze() at the end of the pipeline. Carries
    # only evidence records actually referenced by some artefact (params,
    # assumptions, ProsConsItems). Compression happens at insertion.
    evidence_store: "EvidenceStore | None" = None


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
    """Ranked intervention with predicted lift, cost, and timeline.

    Wave 7: every intervention now carries cost and timeline so ranking
    is by cost-per-adoption-lift rather than raw lift. The
    `cost_usd_low` / `cost_usd_high` pair surfaces the underlying
    estimate uncertainty (the spec's design choice — point cost numbers
    are noisy enough that a range is more honest).
    """

    name: str
    description: str
    adoption_before: float
    adoption_after: float
    lift_pp: float  # percentage points
    mechanism: str
    mechanism_type: str = ""  # free_trial | price_cut | annual_discount | social_proof_push | freemium
    cost_usd_low: float = 0.0
    cost_usd_high: float = 0.0
    timeline_rounds: int = 1
    # USD per percentage point of adoption lift. math.inf when lift_pp <= 0.
    # Primary ranking key — see analyze.py.
    cost_per_adoption_pp: float = float("inf")
    # Optional pairwise-combination metadata (set by rank_interventions when
    # this row represents a combo of two single interventions).
    combined_with: str | None = None
    additivity: str | None = None  # "sub_additive" | "super_additive" | "additive"


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

    # Wave 8 additions — populated by pipeline/kpi.py, segments.py, pros_cons.py.
    # All optional / empty by default so older Wave 0-7 code paths keep working.
    kpi: "KPIDashboard | None" = None
    segments: "SegmentReport | None" = None
    pros_cons: list["ProsConsItem"] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# v2-middle additions (Wave 0): new models wired up in later waves.
# ---------------------------------------------------------------------------


class RoundSnapshot(BaseModel):
    """Per-round state snapshot from the multi-round simulation (Wave 3)."""

    round: int
    phase_counts: dict[str, int] = Field(default_factory=dict)
    cluster_adoption: dict[int, float] = Field(default_factory=dict)
    total_adoption: float = 0.0
    aware_count: int = 0


class EvidenceStrength(BaseModel):
    """Wave 8.5 — counts of supporting evidence for a single claim."""

    n_sim: int = 0
    n_voc: int = 0
    cls: Literal["triangulated", "nuance", "weak"] = "weak"


class ProsConsItem(BaseModel):
    """Triangulated pros/cons claim — requires both simulation and VoC evidence.

    `polarity="nuance"` is used when only one evidence source supports the claim.

    Wave 8.5: `evidence_strength` makes the underlying counts visible.
    `cls` is mechanically derived: n_sim≥1 AND n_voc≥1 → triangulated;
    exactly one side → nuance; zero → weak (validator should block these
    from reaching the narrative author).
    """

    statement: str
    polarity: Literal["pro", "con", "nuance"]
    segments: list[str] = Field(default_factory=list)
    mechanism: str = ""
    simulation_evidence: dict[str, Any] = Field(default_factory=dict)
    voc_evidence: list[str] = Field(default_factory=list)  # quote_ids

    # Wave 8.5.
    evidence_strength: EvidenceStrength = Field(default_factory=EvidenceStrength)


class AdoptionSummary(BaseModel):
    total: float = 0.0
    aware: float = 0.0
    time_to_50pct: int | None = None
    chasm_round: int | None = None


class ForceDominance(BaseModel):
    top_driver: str = ""
    top_blocker: str = ""
    segment_variance: dict[str, float] = Field(default_factory=dict)


class CascadeMetrics(BaseModel):
    first_cluster_crossed_critical_mass: int | None = None
    cluster_spread_rounds: dict[int, int] = Field(default_factory=dict)


class KPIDashboard(BaseModel):
    """Structured KPIs mined from the SimulationResult (Wave 8)."""

    adoption: AdoptionSummary = Field(default_factory=AdoptionSummary)
    force_dominance: ForceDominance = Field(default_factory=ForceDominance)
    convertible_pool: int = 0
    cascade: CascadeMetrics | None = None
    top_sensitivity_params: list[str] = Field(default_factory=list)
    validation_score: float | None = None


class ArchetypeNarrative(BaseModel):
    """Per-archetype adoption story (Wave 8)."""

    archetype: str
    adoption_rate: float
    count: int
    total: int
    dominant_driver: str = ""
    dominant_driver_value: float = 0.0
    dominant_blocker: str = ""
    dominant_blocker_value: float = 0.0
    representative_trajectories: list[str] = Field(default_factory=list)


class SegmentReport(BaseModel):
    """Per-archetype dominant drivers / blockers + sample trajectories."""

    segments: list[ArchetypeNarrative] = Field(default_factory=list)


# Resolve the forward-references in AnalysisReport (kpi/segments/pros_cons)
# and SimulationResult (evidence_store).
AnalysisReport.model_rebuild()
SimulationResult.model_rebuild()
