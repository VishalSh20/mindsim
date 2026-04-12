"""7-Force Decision Engine — Vectorized NumPy.

Computes all 7 behavioral forces on entire agent populations at once.
NO Python loops over agents. Every operation is array-based.

Forces (from ARCHITECTURE.md / RESEARCH.md):
  1. Prospect Value (Kahneman & Tversky 1979/1992)
  2. Reference Price Anchoring (Tversky & Kahneman 1974, Mazumdar 2005)
  3. Status Quo Bias (Samuelson & Zeckhauser 1988)
  4. Social Proof (Cialdini 1984, Salganik 2006)
  5. Anticipated Regret / FOMO (Loomes & Sugden 1982, Zeelenberg 1999)
  6. Hyperbolic Discounting (Laibson 1997, Augenblick 2015)
  7. Identity Signaling (Veblen 1899, Berger & Heath 2007)
"""

from __future__ import annotations

import numpy as np

from mindsim.models.config import SimulationParams


# Prospect theory constants (Tversky & Kahneman 1992)
ALPHA = 0.88  # gain curvature
BETA_PT = 0.88  # loss curvature (distinct from present_bias_beta)
ANCHORING_WEIGHT = 0.3  # anchor effect scaling
IDENTITY_SCALE = 0.2  # identity signaling scaling
DISCOUNT_SCALE = 0.3  # hyperbolic discounting scaling


def compute_forces(
    agents: np.ndarray,
    params: SimulationParams,
) -> dict[str, np.ndarray]:
    """Compute all 7 behavioral forces for every agent.

    Args:
        agents: Structured array with AGENT_DTYPE fields.
        params: Calibrated simulation parameters.

    Returns:
        Dict mapping force name to array of shape (n_agents,).
        Only includes aware agents — unaware agents get NaN.
    """
    n = len(agents)
    aware_mask = agents["aware"]

    # Pre-extract agent fields for readability
    la = agents["loss_aversion_lambda"].astype(np.float64)
    income = agents["income"].astype(np.float64)
    sqb = agents["status_quo_bias"].astype(np.float64)
    spn = agents["social_proof_need"].astype(np.float64)
    fomo_sus = agents["fomo_susceptibility"].astype(np.float64)
    openness = agents["openness"].astype(np.float64)
    comp_frac = agents["competitor_awareness_frac"].astype(np.float64)

    # Pre-extract sim params
    price = params.price
    ref_price_value = params.reference_price.value
    perceived_benefit = params.perceived_benefit.value
    benefit_certainty = params.benefit_certainty.value
    category_penetration = params.category_penetration.value
    switching_cost = params.switching_cost.value
    social_visibility = params.social_visibility.value
    identity_signal = params.identity_signal.value
    time_to_value = params.time_to_value.value
    present_bias_beta = params.present_bias_beta.value
    fomo_intensity = params.fomo_intensity.value
    category_growth = params.category_growth.value
    product_adoption = params.product_adoption_rate.value
    requires_behavior_change = params.requires_behavior_change.value

    # Initialize force arrays with NaN for unaware agents
    forces = {
        "prospect_value": np.full(n, np.nan, dtype=np.float64),
        "anchoring": np.full(n, np.nan, dtype=np.float64),
        "status_quo": np.full(n, np.nan, dtype=np.float64),
        "social_proof": np.full(n, np.nan, dtype=np.float64),
        "fomo": np.full(n, np.nan, dtype=np.float64),
        "hyperbolic_discounting": np.full(n, np.nan, dtype=np.float64),
        "identity_signaling": np.full(n, np.nan, dtype=np.float64),
    }

    if not aware_mask.any():
        return forces

    # Work only on aware agents
    la_a = la[aware_mask]
    income_a = income[aware_mask]
    sqb_a = sqb[aware_mask]
    spn_a = spn[aware_mask]
    fomo_a = fomo_sus[aware_mask]
    open_a = openness[aware_mask]
    comp_frac_a = comp_frac[aware_mask]

    # ═══════════════════════════════════════════════════════════════
    # FORCE 1: PROSPECT VALUE (Kahneman & Tversky 1979/1992)
    #
    # GAIN = perceived_benefit ^ α
    #
    # LOSS = λ × loss_input ^ β, where loss_input combines:
    #   1. Reference-price-relative loss ("is this expensive for what it is?")
    #   2. Absolute subscription pain (log scale, $0=none, $20=moderate, $100+=major)
    #   3. Income relief (high income reduces pain, doesn't eliminate it)
    # ═══════════════════════════════════════════════════════════════
    gain = np.power(np.clip(perceived_benefit, 1e-10, None), ALPHA)

    if price > 0:
        # Component 1: Price relative to reference price
        # price_ratio > 1 means expensive vs reference, < 1 means cheap
        effective_ref = max(ref_price_value, 1.0)
        price_ratio = price / effective_ref
        relative_loss = np.maximum(0.0, (price_ratio - 1.0) / price_ratio)

        # Component 2: Absolute monthly subscription pain (log scale)
        # $0=0, $10≈0.35, $20≈0.46, $50≈0.75, $200≈1.27
        absolute_pain = np.log1p(price / 10.0) / np.log1p(10.0)

        # Component 3: Income reduces pain by up to 40%
        monthly_income = income_a / 12.0
        income_relief = 0.4 * np.minimum(1.0, monthly_income / 15000.0)

        # Combined loss input (0-1+ scale)
        loss_input = (relative_loss * 0.5 + absolute_pain * 0.5) * (1.0 - income_relief)
        loss_input = np.clip(loss_input, 1e-10, None)

        loss = la_a * np.power(loss_input, BETA_PT)
    else:
        # Free product: no loss
        loss = np.zeros_like(la_a)

    prospect = gain - loss
    forces["prospect_value"][aware_mask] = prospect

    # ═══════════════════════════════════════════════════════════════
    # FORCE 2: REFERENCE PRICE ANCHORING (Tversky & Kahneman 1974)
    # ref = awareness-weighted avg of known competitor prices
    # anchor_effect = (ref - price) / max(ref, 1) × 0.3
    # ═══════════════════════════════════════════════════════════════
    # Per-agent reference price: modulated by their competitor awareness
    # Innovators know more competitors → their ref reflects full market
    # Laggards know fewer → their ref is skewed toward dominant player
    if ref_price_value > 0 and price > 0:
        # Scale reference price by competitor awareness
        # Less aware agents anchor more to the dominant/cheapest option
        agent_ref = ref_price_value * comp_frac_a + price * (1.0 - comp_frac_a)
        anchor = (agent_ref - price) / np.maximum(agent_ref, 1.0) * ANCHORING_WEIGHT
    else:
        anchor = np.zeros(la_a.shape)

    forces["anchoring"][aware_mask] = anchor

    # ═══════════════════════════════════════════════════════════════
    # FORCE 3: STATUS QUO BIAS (Samuelson & Zeckhauser 1988)
    #
    # The status quo is NOT "using a competitor" — it's "whatever the
    # person currently does, including nothing."
    #
    # (1-penetration) fraction use nothing → barrier = adopting new behavior
    # (penetration) fraction use a tool → barrier = switching products
    #
    # Low category penetration → MOST people face behavior-change barrier
    # (the harder kind), not just a product switch.
    # ═══════════════════════════════════════════════════════════════
    # People NOT in the category: adopting a whole new behavior
    adoption_friction = 0.5 + 0.5 * requires_behavior_change

    # People IN the category: switching between products
    switching_friction = 0.3 + 0.7 * switching_cost

    # Blend based on penetration:
    # Low penetration (0.10) → 90% face adoption_friction, 10% face switching
    blended_friction = (
        (1.0 - category_penetration) * adoption_friction
        + category_penetration * switching_friction
    )

    status_quo = -(sqb_a * blended_friction)
    forces["status_quo"][aware_mask] = status_quo

    # ═══════════════════════════════════════════════════════════════
    # FORCE 4: SOCIAL PROOF (Cialdini 1984, Salganik 2006)
    # social_proof_need × log(1 + adoption × visibility) × (1 - benefit_certainty)
    # Matters MORE when benefit is uncertain.
    # ═══════════════════════════════════════════════════════════════
    social = (
        spn_a
        * np.log1p(product_adoption * social_visibility)
        * (1.0 - benefit_certainty)
    )
    forces["social_proof"][aware_mask] = social

    # ═══════════════════════════════════════════════════════════════
    # FORCE 5: ANTICIPATED REGRET / FOMO (Loomes & Sugden 1982)
    # fomo_intensity × social_visibility × category_growth × social_proof_need
    # Distinct from loss aversion — this is opportunity-miss pain.
    # ═══════════════════════════════════════════════════════════════
    fomo = fomo_intensity * social_visibility * category_growth * fomo_a
    forces["fomo"][aware_mask] = fomo

    # ═══════════════════════════════════════════════════════════════
    # FORCE 6: HYPERBOLIC DISCOUNTING (Laibson 1997, Augenblick 2015)
    # -(time_to_value × (1 - β_agent) × perceived_benefit × 0.3)
    # β ∈ {0.5 for behavior-change, 0.85 for consumption}
    #
    # Per-agent variation per Frederick et al. 2002: discount rates
    # vary enormously across individuals. We modulate the product-level
    # β by agent openness: high-openness agents (novelty seekers) have
    # weaker present bias (β closer to 1.0), low-openness agents have
    # stronger present bias (β closer to 0.0). This produces ±15%
    # variation around the calibrated β.
    # ═══════════════════════════════════════════════════════════════
    # Per-agent beta: openness shifts beta toward 1.0 (less bias)
    # openness=0.5 → no shift; openness=1.0 → beta + 0.15; openness=0 → beta - 0.15
    agent_beta = present_bias_beta + 0.3 * (open_a - 0.5)
    agent_beta = np.clip(agent_beta, 0.1, 0.95)

    discount = -(
        time_to_value * (1.0 - agent_beta) * perceived_benefit * DISCOUNT_SCALE
    )
    forces["hyperbolic_discounting"][aware_mask] = discount

    # ═══════════════════════════════════════════════════════════════
    # FORCE 7: IDENTITY SIGNALING (Veblen 1899, Berger & Heath 2007)
    # openness × identity_signal × social_visibility × 0.2
    # ═══════════════════════════════════════════════════════════════
    identity = open_a * identity_signal * social_visibility * IDENTITY_SCALE
    forces["identity_signaling"][aware_mask] = identity

    return forces


def compute_decisions(
    forces: dict[str, np.ndarray],
    temperature: float = 3.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute adoption decisions from force totals.

    Args:
        forces: Dict of force arrays from compute_forces.
        temperature: Logistic function steepness parameter.
        rng: Random number generator.

    Returns:
        (adopt_prob, decisions) — both shape (n_agents,).
        Unaware agents get prob=0 and decision=False.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get length from any force array
    n = len(next(iter(forces.values())))

    # Sum all forces
    total_utility = np.zeros(n, dtype=np.float64)
    for force_arr in forces.values():
        # NaN (unaware) agents contribute 0
        total_utility += np.nan_to_num(force_arr, nan=0.0)

    # Identify aware agents (those with non-NaN in any force)
    aware_mask = ~np.isnan(next(iter(forces.values())))

    # Logistic function: P(adopt) = 1 / (1 + exp(-utility × temperature))
    adopt_prob = np.zeros(n, dtype=np.float64)
    x = total_utility[aware_mask] * temperature
    # Clip to prevent overflow
    x = np.clip(x, -500.0, 500.0)
    adopt_prob[aware_mask] = 1.0 / (1.0 + np.exp(-x))

    # Stochastic decisions
    decisions = np.zeros(n, dtype=np.bool_)
    random_draws = rng.random(aware_mask.sum())
    decisions[aware_mask] = random_draws < adopt_prob[aware_mask]

    return adopt_prob, decisions


def compute_weighted_forces(
    forces: dict[str, np.ndarray],
    adopt_prob: np.ndarray,
) -> dict[str, float]:
    """Compute boundary-weighted force averages.

    Agents near prob=0.5 (the "convertible pool") contribute more,
    emphasizing the forces that actually matter for strategy.

    weight = 1 - (2 × |prob - 0.5|)²

    Args:
        forces: Dict of force arrays.
        adopt_prob: Adoption probability for each agent.

    Returns:
        Dict mapping force name to weighted average value.
    """
    # Compute boundary weights: agents near 0.5 get weight ~1.0
    # Agents near 0 or 1 get weight ~0.0
    weights = 1.0 - (2.0 * np.abs(adopt_prob - 0.5)) ** 2
    weights = np.maximum(weights, 0.0)

    # Only consider aware agents
    aware_mask = adopt_prob > 0  # unaware agents have prob=0

    total_weight = weights[aware_mask].sum()
    if total_weight < 1e-10:
        total_weight = 1.0  # avoid division by zero

    result = {}
    for name, force_arr in forces.items():
        valid = ~np.isnan(force_arr)
        combined_mask = aware_mask & valid
        weighted_sum = (force_arr[combined_mask] * weights[combined_mask]).sum()
        result[name] = float(weighted_sum / total_weight)

    return result
