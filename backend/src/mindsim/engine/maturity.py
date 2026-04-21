"""Category-maturity helpers.

Wave 1 shipped the `maturity → consideration_threshold` mapping.
Wave 5 adds `classify_from_signals()` — a pytrends-free heuristic that
derives maturity from scraped corpus signals (competitor count, review
volume, optional category-age hint).

Plan reference: v2-middle §R4 / §6.4 / WAVES-4-TO-8 §Wave 5.
"""
from __future__ import annotations

from typing import Literal

Maturity = Literal["nascent", "growing", "mainstream", "saturated"]

# Thresholds per MECHANICS-v2 §AWARE→CONSIDERING.
_CONSIDERATION_THRESHOLDS: dict[str, float] = {
    "nascent": 0.60,
    "growing": 0.35,
    "mainstream": 0.45,
    "saturated": 0.50,
}

# Used when maturity is unknown/unclassified. Midpoint of the range.
_DEFAULT_CONSIDERATION_THRESHOLD = 0.45


def consideration_threshold_for_maturity(maturity: str | None) -> float:
    """Return the awareness_strength threshold for entering consideration.

    Args:
        maturity: One of "nascent", "growing", "mainstream", "saturated",
            or None/unknown.

    Returns:
        Threshold in [0, 1].
    """
    if maturity is None:
        return _DEFAULT_CONSIDERATION_THRESHOLD
    return _CONSIDERATION_THRESHOLDS.get(maturity, _DEFAULT_CONSIDERATION_THRESHOLD)


# ─────────────── Wave 5: heuristic maturity classifier ───────────────


def classify_from_signals(
    n_competitors: int,
    review_volume: int,
    category_age_hint: str | None = None,
) -> Maturity:
    """Classify category maturity from research signals (no pytrends).

    Inputs come from the research stage:
      - n_competitors: count of verified + discovered competitors.
      - review_volume: total scraped VoC document count (Reddit + HN +
        review sites). Proxy for "how much chatter exists".
      - category_age_hint: optional LLM-derived hint ("new in 2024",
        "established 1990s") used as tie-breaker.

    Bands (calibrated against canonical products):
      - nascent     : ≤ 2 competitors AND review_volume < 100
      - growing     : ≤ 5 competitors AND review_volume < 5_000
      - mainstream  : > 5 competitors AND review_volume < 50_000
      - saturated   : review_volume ≥ 50_000

    The age hint tilts borderline cases (e.g. "established" pushes a
    growing category to mainstream).
    """
    n_competitors = max(0, int(n_competitors))
    review_volume = max(0, int(review_volume))

    if review_volume >= 50_000:
        bucket: Maturity = "saturated"
    elif n_competitors <= 2 and review_volume < 100:
        bucket = "nascent"
    elif n_competitors <= 5 and review_volume < 5_000:
        bucket = "growing"
    else:
        bucket = "mainstream"

    if category_age_hint:
        hint = category_age_hint.lower()
        if bucket == "growing" and any(
            kw in hint for kw in ("established", "decades", "1990", "1980", "legacy")
        ):
            bucket = "mainstream"
        elif bucket == "mainstream" and any(
            kw in hint for kw in ("new", "emerging", "2024", "2025", "2026")
        ):
            bucket = "growing"

    return bucket
