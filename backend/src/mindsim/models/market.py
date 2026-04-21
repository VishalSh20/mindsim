"""MarketContext — research results with confidence flags."""

from pydantic import BaseModel, ConfigDict, Field


class VerifiedCompetitor(BaseModel):
    """A competitor whose data was verified via research.

    Wave 5 additions: `feature_scores` (per-category strength 0-1),
    `market_share` (fractional 0-1), `source_ids` (stable src:<hash> refs
    back to research documents for Wave 8.5 EvidenceStore).
    """

    name: str
    confirmed_price: float | None = None
    price_model: str | None = None
    has_free_tier: bool = False
    market_position: str | None = None
    source_url: str | None = None

    # Wave 5 — synthesized by A2.
    feature_scores: dict[str, float] = Field(default_factory=dict)
    market_share: float | None = None
    source_ids: list[str] = Field(default_factory=list)


class Cluster(BaseModel):
    """A market cluster — a subgroup of agents who share network exposure.

    Wave 4 uses a deterministic 6-cluster fallback scheme (archetype-tier
    x income-tier). Wave 5 replaces population from scraped communities.
    `visibility_factor` multiplies social_visibility in cluster-local
    social proof and the WOM term in awareness promotion.
    """

    id: int
    name: str
    visibility_factor: float
    size: int = 0


class MarketContext(BaseModel):
    """Output of the research stage — verified market data."""

    verified_competitors: list[VerifiedCompetitor] = Field(default_factory=list)
    discovered_competitors: list[VerifiedCompetitor] = Field(default_factory=list)

    category_penetration: float | None = None
    category_penetration_confidence: float = 0.3

    category_growth: float | None = None  # from Google Trends
    category_maturity: str | None = None  # nascent | growing | mainstream | saturated

    # v2-middle Wave 4: market clusters. Populated after simulate() builds
    # the agent population. In Wave 5, cluster discovery becomes scraper-
    # driven.
    clusters: list[Cluster] = Field(default_factory=list)

    # v2-middle Wave 5 — A2/A3 artefacts surfaced for downstream stages.
    # `voc_report` is optional (import avoided here to dodge a cycle).
    voc_report: object | None = None
    # Raw evidence buffers retained for Wave 8.5 EvidenceStore assembly.
    tavily_results_cache: list[dict] = Field(default_factory=list)
    scraped_docs_cache: list[object] = Field(default_factory=list)

    low_confidence_flags: list[str] = Field(default_factory=list)
    research_cost: int = 0  # total Tavily credits spent
    research_skipped: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def empty(cls) -> "MarketContext":
        """Return an empty MarketContext for --skip-research mode."""
        return cls(research_skipped=True)
