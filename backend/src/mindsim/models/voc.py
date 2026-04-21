"""Voice-of-Customer models.

Scaffolded in Wave 0, populated in Wave 5 by the A3 VoC Analyst agent.
`ScrapedDocument` lives in `mindsim.scrape.base` and is re-exported here
for convenience.

Wave 5 additions:
  - `VoCQuote` with stable `voc:<hash>` id (anchors Wave 8.5 EvidenceStore).
  - `FeatureSentiment` gains `net_sentiment` + `n_mentions` for synthesizer use.
  - `VoCReport.quotes` + `source_breakdown` + `corpus_size`.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mindsim.scrape.base import ScrapedDocument

__all__ = [
    "FeatureSentiment",
    "VoCQuote",
    "VoCReport",
    "ScrapedDocument",
]

Polarity = Literal["pain", "delight", "neutral"]
Archetype = Literal[
    "innovator", "early_adopter", "early_majority",
    "late_majority", "laggard", "unknown",
]


class VoCQuote(BaseModel):
    """One voice-of-customer excerpt with stable ID."""

    quote_id: str  # "voc:<hash>" — matches ScrapedDocument.quote_id.
    text: str
    polarity: Polarity
    archetype_hint: Archetype = "unknown"
    source: Literal["reddit", "hackernews", "pricing_page", "review", "other"] = "other"
    url: str | None = None


class FeatureSentiment(BaseModel):
    """Sentiment split for a single feature dimension."""

    feature: str
    positive_pct: float = 0.0  # 0-1
    negative_pct: float = 0.0  # 0-1
    quote_ids: list[str] = Field(default_factory=list)

    # Wave 5: summary metrics used by the synthesizer.
    net_sentiment: float = 0.0     # positive_pct - negative_pct
    n_mentions: int = 0


class VoCReport(BaseModel):
    """Synthesized voice-of-customer output from A3.

    `bias_note` is REQUIRED when sources are Reddit/HN-weighted. Narrative
    author must surface this when pros/cons lean on VoC.
    """

    pain_points: list[str] = Field(default_factory=list)
    delight_points: list[str] = Field(default_factory=list)
    feature_sentiment: list[FeatureSentiment] = Field(default_factory=list)
    complaint_themes: list[str] = Field(default_factory=list)
    unmet_needs: list[str] = Field(default_factory=list)
    quotes: list[VoCQuote] = Field(default_factory=list)
    bias_note: str | None = None
    source_count: int = 0
    corpus_size: int = 0  # alias for source_count — Wave 5 explicit name
    source_breakdown: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def empty(cls) -> "VoCReport":
        return cls(
            bias_note="No scraped VoC available for this product.",
            source_count=0,
            corpus_size=0,
        )
