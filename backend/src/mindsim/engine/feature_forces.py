"""Per-feature gain/loss computation — the v2-middle Wave 2 prospect path.

When a SimulationParams.feature_matrix is populated (by A4), prospect
value is computed by summing over features:

    gain_i  = Σ_{f : polarity=+} agent_w[i, f.category] × f.score × w(f.certainty)
    loss_i  = Σ_{f : polarity=-} agent_w[i, f.category] × f.score × w(f.certainty)

`w(p)` is the Tversky-Kahneman probability weighting function already
implemented in engine/probability_weighting.py.

Time-to-value and visibility are aggregated the same way but use a
weighted mean rather than a weighted sum, so their output sits on the
same [0, 1]-ish scale the existing force code expects.

All functions are vectorized over the agent array — zero Python loops
over agents.
"""
from __future__ import annotations

import numpy as np

from mindsim.engine.feature_weights import FEATURE_CATEGORIES
from mindsim.engine.probability_weighting import DEFAULT_GAMMA, probability_weight
from mindsim.models.product import Feature

# Map from feature category → AGENT_DTYPE field holding that category's
# per-agent weight. Kept in sync with engine/population.py AGENT_DTYPE.
CATEGORY_TO_WEIGHT_FIELD = {
    c: f"feature_weight_{c}" for c in FEATURE_CATEGORIES
}


def _agent_weights_for(agents: np.ndarray, feature: Feature) -> np.ndarray:
    field = CATEGORY_TO_WEIGHT_FIELD[feature.category]
    return agents[field].astype(np.float64)


def compute_feature_gain(
    agents: np.ndarray,
    features: list[Feature],
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    """Per-agent gain from positive-polarity features.

    `agents` is a structured array (already sliced to whatever subset
    the caller wants scored — e.g. aware agents only). Returns an array
    of the same length with the weighted, probability-weighted gain.
    """
    n = len(agents)
    gain = np.zeros(n, dtype=np.float64)
    for feat in features:
        if feat.polarity != "positive":
            continue
        w_cat = _agent_weights_for(agents, feat)
        weighted_cert = float(probability_weight(float(feat.certainty), gamma=gamma))
        gain += w_cat * float(feat.score) * weighted_cert
    return gain


def compute_feature_loss(
    agents: np.ndarray,
    features: list[Feature],
    gamma: float = DEFAULT_GAMMA,
) -> np.ndarray:
    """Per-agent loss from negative-polarity features.

    Does NOT include monetary (price) loss — that stays in forces.py
    where it already lives. This is specifically the ongoing-cost /
    friction loss introduced by the feature matrix.
    """
    n = len(agents)
    loss = np.zeros(n, dtype=np.float64)
    for feat in features:
        if feat.polarity != "negative":
            continue
        w_cat = _agent_weights_for(agents, feat)
        weighted_cert = float(probability_weight(float(feat.certainty), gamma=gamma))
        loss += w_cat * float(feat.score) * weighted_cert
    return loss


def compute_weighted_time_to_value_months(
    agents: np.ndarray,
    features: list[Feature],
) -> np.ndarray:
    """Per-agent weighted mean time-to-value in months (over positive features).

    Agents who weight `core_value` heavily are delayed by a slow
    core_value feature; agents who don't weight it heavily aren't.
    """
    n = len(agents)
    pos = [f for f in features if f.polarity == "positive"]
    if not pos:
        return np.zeros(n, dtype=np.float64)

    total_weight = np.zeros(n, dtype=np.float64)
    weighted_ttv = np.zeros(n, dtype=np.float64)
    for feat in pos:
        w_cat = _agent_weights_for(agents, feat)
        total_weight += w_cat
        weighted_ttv += w_cat * float(feat.time_to_value_months)
    return np.where(total_weight > 1e-12, weighted_ttv / np.maximum(total_weight, 1e-12), 0.0)


def compute_weighted_visibility(
    agents: np.ndarray,
    features: list[Feature],
    fallback: float = 0.3,
) -> np.ndarray:
    """Per-agent weighted visibility from positive features.

    Falls back to a mild default when no positive features exist
    (should be rare — caught earlier by the validator).
    """
    n = len(agents)
    pos = [f for f in features if f.polarity == "positive"]
    if not pos:
        return np.full(n, fallback, dtype=np.float64)

    total_weight = np.zeros(n, dtype=np.float64)
    weighted_vis = np.zeros(n, dtype=np.float64)
    for feat in pos:
        w_cat = _agent_weights_for(agents, feat)
        total_weight += w_cat
        weighted_vis += w_cat * float(feat.visibility)
    return np.where(
        total_weight > 1e-12,
        weighted_vis / np.maximum(total_weight, 1e-12),
        fallback,
    )


def compute_weighted_certainty(
    agents: np.ndarray,
    features: list[Feature],
) -> np.ndarray:
    """Per-agent weighted *unweighted* certainty (for use in social proof).

    Social proof scales with (1 − w(certainty)) — see forces.py Force 4.
    We return the raw certainty here so the social proof formula still
    applies w() at its point of use.
    """
    n = len(agents)
    if not features:
        return np.full(n, 0.5, dtype=np.float64)

    total_weight = np.zeros(n, dtype=np.float64)
    weighted_cert = np.zeros(n, dtype=np.float64)
    for feat in features:
        w_cat = _agent_weights_for(agents, feat)
        total_weight += w_cat
        weighted_cert += w_cat * float(feat.certainty)
    return np.where(
        total_weight > 1e-12,
        weighted_cert / np.maximum(total_weight, 1e-12),
        0.5,
    )
