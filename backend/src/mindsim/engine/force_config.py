"""Force-interaction constants.

Centralises the thresholds that govern *how* forces combine — not what
they evaluate to for a given agent. These are modelling decisions backed
by the research docs, deliberately kept out of the LLM calibration path
so the calibrator cannot wiggle them.

Each constant has a citation or a v2-middle §reference.
"""
from __future__ import annotations

# ─── Loss × Discount multiplicative rule (Gap 5 / v2-middle §R10) ───
#
# RESEARCH.md: "loss_aversion × hyperbolic_discounting multiply for
# products with upfront cost + delayed benefit". The condition fires
# only when BOTH of the following hold:
#
#   requires_behavior_change > BEHAVIOR_CHANGE_THRESHOLD
#   time_to_value            > DELAYED_BENEFIT_THRESHOLD
#
# `time_to_value` is normalised to [0, 1] in v1, not raw months. 0.30
# corresponds roughly to "more than a few months to realise value" given
# how the calibrator scales this parameter.
BEHAVIOR_CHANGE_THRESHOLD = 0.5
DELAYED_BENEFIT_THRESHOLD = 0.30

# Multiplicative penalty intensity when the rule fires.
# Formula: effective_prospect = prospect * (1 - DELAYED_PENALTY_WEIGHT * discount_magnitude)
# Chosen so that for the canonical habit-change product the penalty is
# comparable to the additive discount, but the penalty scales with the
# prospect magnitude (that's what "multiplicative" means).
DELAYED_PENALTY_WEIGHT = 1.0


# ─── Reference-price anchoring absorption (Gap 7 / v2-middle §R1) ───
#
# The standalone anchoring force is deleted. The effect is re-expressed
# as a multiplicative scaling on prospect loss:
#
#   loss *= clip(price / agent_ref_price, REF_RATIO_FLOOR, REF_RATIO_CEIL)
#
# - price < agent_ref: ratio < 1 → loss shrinks (product feels cheap
#   relative to what I know). Captures the v1 "anchor bonus" without
#   double-counting it as an additive force.
# - price > agent_ref: ratio > 1 → loss grows (expensive surprise).
# - ratio == 1: no change (agent has no meaningful anchor).
REF_RATIO_FLOOR = 0.30
REF_RATIO_CEIL = 2.00
