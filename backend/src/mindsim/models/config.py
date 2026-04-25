"""SimulationConfig — all numerical parameters with basis and confidence.

This is the ONLY place numerical simulation parameters live.
Each parameter has a value, basis (why this number), and confidence (how sure).
Optionally, parameters can have per-archetype overrides (by_archetype).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mindsim.models.product import Feature

# Wave 8.5 — provenance tags. Same Literal as models/evidence.SourceType,
# duplicated here to avoid an import cycle (evidence.py is imported by
# downstream consumers that also need config.py).
ParamSourceType = Literal["voc", "research", "trends", "default", "llm_judgment"]


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
    """A single calibrated parameter with provenance.

    Wave 8.5: optional `evidence_refs` carry IDs that resolve in the
    `EvidenceStore` attached to a SimulationResult. Empty list (the
    default) means "no resolvable evidence" — the validator flags this
    as sparse-rationale unless `source_type` is `"default"`.
    """

    value: float
    basis: str = ""  # why this number
    confidence: float = 0.5  # 0-1, how confident we are
    by_archetype: dict[str, float] | None = None  # optional per-archetype overrides

    # Wave 8.5 additions — optional, default to empty so existing
    # code paths and serialised v0..v8 dumps still parse cleanly.
    evidence_refs: list[str] = Field(default_factory=list)
    source_type: ParamSourceType | None = None


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
    """A challengeable assumption with an ID for the user.

    Wave 8.5: `evidence_refs` link into the run's EvidenceStore. The
    validator rejects assumptions with `source_type != "default"` and
    no resolvable refs (sparse-rationale). `published_bounds` is
    populated from RESEARCH.md-derived constants for parameters with
    published ranges (e.g. λ ∈ [1.0, 4.0]); validator rejects values
    outside the band.
    """

    id: str  # "A1", "A2", etc.
    parameter: str
    value: float
    basis: str
    confidence: float
    sensitivity: str = "unknown"  # "high", "medium", "low"

    # Wave 8.5 additions.
    evidence_refs: list[str] = Field(default_factory=list)
    source_type: ParamSourceType | None = None
    published_bounds: tuple[float, float] | None = None


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

    # v2-middle Wave 3: multi-round simulation rounds.
    n_rounds: int = 8

    # v2-middle Wave 2: product as feature vectors.
    # Populated by A4 (calibrate stage). When present, the feature-matrix
    # path in engine/feature_forces.py replaces the scalar prospect math.
    # Empty by default — pipelines that haven't yet rolled to Wave 2 keep
    # using the scalar path.
    feature_matrix: list[Feature] = Field(default_factory=list)

    # Raw competitor feature scores as emitted by A4. Keyed by competitor name.
    # Wave 2 stores them here as placeholders; Wave 4 scrapers replace them
    # with grounded values and populate CompetitorInfo.feature_scores on the
    # actual competitor objects.
    competitor_feature_scores: dict[str, dict[str, float]] = Field(default_factory=dict)

    # v2-middle Wave 7: free-trial intervention machinery.
    # When `free_trial_active` is True, the state machine routes a fraction
    # of CONSIDERING agents into TRIALING for `free_trial_duration_rounds`
    # rounds before they decide whether to convert. Defaults off so non-
    # intervention sims behave exactly as Waves 0-6.
    free_trial_active: bool = False
    free_trial_duration_rounds: int = 3
    free_trial_entry_threshold: float = 0.20  # min adopt_prob to enter trial


class SimulationConfig(BaseModel):
    """Complete simulation config — output of the calibrate stage."""

    simulation_params: SimulationParams = Field(default_factory=SimulationParams)
    population_config: PopulationConfig = Field(default_factory=PopulationConfig)
    assumptions: list[Assumption] = Field(default_factory=list)
