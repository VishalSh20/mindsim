"""Tversky-Kahneman probability weighting.

Implements `w(p) = p^γ / (p^γ + (1 − p)^γ)^(1/γ)`, the probability weighting
function fitted in Tversky & Kahneman (1992) with γ=0.61 for gains.

Under this weighting:
  - Small probabilities are overweighted (w(0.1) ≈ 0.19 > 0.1).
  - Large probabilities are underweighted (w(0.9) ≈ 0.71 < 0.9).
  - The crossover point (w(p) = p) sits near p ≈ 1/3 for γ=0.61.

This is the *probability* weighting function, distinct from the *value*
function (v(x) = x^α for gains, -λ|x|^β for losses). Use this on any
quantity that plays the role of a subjective probability: benefit
certainty, trial success chance, adoption expectation, etc.
"""
from __future__ import annotations

import numpy as np

DEFAULT_GAMMA = 0.61  # Tversky & Kahneman 1992, fitted for gains


def probability_weight(
    p: float | np.ndarray,
    gamma: float = DEFAULT_GAMMA,
) -> float | np.ndarray:
    """Apply the T&K probability weighting function.

    Vectorized. Inputs are clipped to [0, 1] to guarantee w(0) = 0, w(1) = 1
    and to tolerate upstream calibration noise.

    Args:
        p: Probability (or array of probabilities). Clipped to [0, 1].
        gamma: Curvature parameter. 0.61 for gains (T&K 1992),
            0.69 for losses. Must be in (0, 1].

    Returns:
        Weighted probability, same shape as input.
    """
    if not 0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma}")

    p_arr = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)

    p_pow = np.power(p_arr, gamma)
    q_pow = np.power(1.0 - p_arr, gamma)
    denom = np.power(p_pow + q_pow, 1.0 / gamma)

    # Endpoints: p=0 → numerator=0; p=1 → denominator = numerator = 1.
    # Guard against 0/0 explicitly (only at p=0 with gamma<1 giving 0/0 numerically stable = 0).
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denom > 0.0, p_pow / denom, 0.0)

    # Force exact endpoints.
    result = np.where(p_arr <= 0.0, 0.0, result)
    result = np.where(p_arr >= 1.0, 1.0, result)

    if np.isscalar(p) or (isinstance(p, np.ndarray) and p.ndim == 0):
        return float(result)
    return result
