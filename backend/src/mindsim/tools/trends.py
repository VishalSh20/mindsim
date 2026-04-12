"""Google Trends wrapper — category growth and maturity signals."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_category_trends(keywords: list[str]) -> dict:
    """Get trend data for category keywords.

    Args:
        keywords: List of search terms (max 5 for pytrends).

    Returns:
        Dict with growth_rate, interest, maturity classification.
    """
    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360)
        kw_list = keywords[:5]  # pytrends max 5

        pytrends.build_payload(kw_list, timeframe="today 12-m")
        interest_df = pytrends.interest_over_time()

        if interest_df.empty:
            logger.warning("Google Trends returned empty data")
            return _default_trends()

        # Average interest across keywords
        cols = [c for c in interest_df.columns if c != "isPartial"]
        avg_interest = interest_df[cols].mean(axis=1)

        # Current interest (last value)
        current_interest = float(avg_interest.iloc[-1])

        # Growth rate (slope of last 6 months vs first 6 months)
        n = len(avg_interest)
        if n > 12:
            first_half = avg_interest.iloc[: n // 2].mean()
            second_half = avg_interest.iloc[n // 2 :].mean()
            if first_half > 0:
                growth_rate = (second_half - first_half) / first_half
            else:
                growth_rate = 0.0
        else:
            growth_rate = 0.0

        # Maturity classification
        maturity = _classify_maturity(current_interest, growth_rate)

        return {
            "growth_rate": float(growth_rate),
            "interest": float(current_interest),
            "maturity": maturity,
            "keywords_used": kw_list,
        }

    except Exception as e:
        logger.warning(f"Google Trends failed: {e}")
        return _default_trends()


def _classify_maturity(interest: float, growth_rate: float) -> str:
    """Classify category maturity from trends data.

    Per ARCHITECTURE.md:
      nascent (<20 interest)
      growing (20-60, positive slope)
      mainstream (60+, flat)
      declining (negative slope)
    """
    if interest < 20:
        return "nascent"
    elif growth_rate < -0.1:
        return "declining"
    elif interest >= 60 and abs(growth_rate) < 0.1:
        return "mainstream"
    elif growth_rate > 0:
        return "growing"
    else:
        return "mainstream"


def _default_trends() -> dict:
    """Fallback trend data when Trends API fails."""
    return {
        "growth_rate": 0.0,
        "interest": 50.0,
        "maturity": "growing",
        "keywords_used": [],
    }
