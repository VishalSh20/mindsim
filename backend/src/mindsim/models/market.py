"""MarketContext — research results with confidence flags."""

from pydantic import BaseModel, Field


class VerifiedCompetitor(BaseModel):
    """A competitor whose data was verified via research."""

    name: str
    confirmed_price: float | None = None
    price_model: str | None = None
    has_free_tier: bool = False
    market_position: str | None = None
    source_url: str | None = None


class MarketContext(BaseModel):
    """Output of the research stage — verified market data."""

    verified_competitors: list[VerifiedCompetitor] = Field(default_factory=list)
    discovered_competitors: list[VerifiedCompetitor] = Field(default_factory=list)

    category_penetration: float | None = None
    category_penetration_confidence: float = 0.3

    category_growth: float | None = None  # from Google Trends
    category_maturity: str | None = None  # nascent | growing | mainstream | saturated

    low_confidence_flags: list[str] = Field(default_factory=list)
    research_cost: int = 0  # total Tavily credits spent
    research_skipped: bool = False

    @classmethod
    def empty(cls) -> "MarketContext":
        """Return an empty MarketContext for --skip-research mode."""
        return cls(research_skipped=True)
