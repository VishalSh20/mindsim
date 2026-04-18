"""Category-maturity helpers.

Wave 1 ships the `maturity → consideration_threshold` mapping only.
The classifier that *derives* maturity from scraped signals arrives in
Wave 5 (replacing Google Trends per v2-middle §R4 / §6.4).

Plan reference: §5.5. The threshold modulates how easily aware agents
enter the consideration phase. It isn't consumed yet — that's Wave 3
state machine — but it's computed here so downstream waves can read it
without refactoring the calibration output shape.
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
