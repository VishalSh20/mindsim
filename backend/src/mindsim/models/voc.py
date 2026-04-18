"""Voice-of-Customer models (Wave 5 artifact, scaffolded in Wave 0).

Populated by the A3 VoC Analyst agent from scraped documents.
`ScrapedDocument` lives in `mindsim.scrape.base` and is re-exported here
for convenience.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from mindsim.scrape.base import ScrapedDocument

__all__ = ["FeatureSentiment", "VoCReport", "ScrapedDocument"]


class FeatureSentiment(BaseModel):
    """Sentiment split for a single feature dimension."""

    feature: str
    positive_pct: float = 0.0  # 0-1
    negative_pct: float = 0.0  # 0-1
    quote_ids: list[str] = Field(default_factory=list)


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
    bias_note: str | None = None
    source_count: int = 0
