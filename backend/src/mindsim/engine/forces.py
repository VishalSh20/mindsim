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
    conscien = agents["conscientiousness"].astype(np.float64)
    novelty_w = agents["novelty_weight"].astype(np.float64)
    price_sens = agents["price_sensitivity"].astype(np.float64)

    # Pre-extract sim params (scalar fallbacks)
    price = params.price
    identity_signal = params.identity_signal.value
    present_bias_beta = params.present_bias_beta.value
    fomo_intensity = params.fomo_intensity.value
    category_growth = params.category_growth.value
    product_adoption = params.product_adoption_rate.value
    requires_behavior_change = params.requires_behavior_change.value
    category_penetration = params.category_penetration.value

    # 6 product-perception params: per-agent arrays if stamped, else scalar fallback.
    # When per-agent, these are full-population arrays that get [aware_mask]-sliced below.
    has_agent_params = (
        "agent_perceived_benefit" in agents.dtype.names
        and agents["agent_perceived_benefit"].any()
    )
    if has_agent_params:
        perceived_benefit = agents["agent_perceived_benefit"].astype(np.float64)
        benefit_certainty = agents["agent_benefit_certainty"].astype(np.float64)
        switching_cost = agents["agent_switching_cost"].astype(np.float64)
        ref_price_value = agents["agent_reference_price"].astype(np.float64)
        social_visibility = agents["agent_social_visibility"].astype(np.float64)
        time_to_value = agents["agent_time_to_value"].astype(np.float64)
    else:
        perceived_benefit = params.perceived_benefit.value
        benefit_certainty = params.benefit_certainty.value
        switching_cost = params.switching_cost.value
        ref_price_value = params.reference_price.value
        social_visibility = params.social_visibility.value
        time_to_value = params.time_to_value.value

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
    conscien_a = conscien[aware_mask]
    novelty_a = novelty_w[aware_mask]
    price_sens_a = price_sens[aware_mask]

    # Slice per-agent product params to aware subset (arrays) or keep scalar
    if has_agent_params:
        perceived_benefit_a = perceived_benefit[aware_mask]
        benefit_certainty_a = benefit_certainty[aware_mask]
        switching_cost_a = switching_cost[aware_mask]
        ref_price_a = ref_price_value[aware_mask]
        social_visibility_a = social_visibility[aware_mask]
        time_to_value_a = time_to_value[aware_mask]
    else:
        perceived_benefit_a = perceived_benefit  # scalar, broadcasts naturally
        benefit_certainty_a = benefit_certainty
        switching_cost_a = switching_cost
        ref_price_a = ref_price_value
        social_visibility_a = social_visibility
        time_to_value_a = time_to_value

    # ═══════════════════════════════════════════════════════════════
    # FORCE 1: PROSPECT VALUE (Kahneman & Tversky 1979/1992)
    #
    # Agent-modulated on BOTH sides:
    #
    # GAIN varies by agent need intensity, category affinity, and
    # novelty seeking — different agents value the same product
    # differently based on personality and situation.
    #
    # LOSS varies by agent income (price as fraction of discretionary
    # income), loss aversion λ, and price sensitivity. For free
    # products, loss is effort-based (switching, learning, habit change)
    # modulated by openness and loss aversion.
    # ═══════════════════════════════════════════════════════════════

    # ── GAIN SIDE — per-agent perceived value ──

    # 1. Need intensity: how badly does THIS agent want this category?
    #    Conscientiousness → values productivity/organization tools
    #    Openness → receptive to trying new products
    #    Competitor awareness → already has alternatives, less excited
    need_intensity = (
        0.5
        + 0.3 * conscien_a
        + 0.2 * open_a
        - 0.3 * comp_frac_a
    )
    need_intensity = np.clip(need_intensity, 0.2, 1.0)

    # 2. Category affinity: would this agent use the product regularly?
    category_affinity = (
        0.4
        + 0.3 * conscien_a
        + 0.3 * (1.0 - price_sens_a)
    )
    category_affinity = np.clip(category_affinity, 0.3, 1.0)

    # 3. Novelty premium: novelty seekers get extra value from new things
    novelty_premium = novelty_a * category_growth * 0.5

    # Agent-specific perceived gain
    agent_gain = perceived_benefit_a * need_intensity * category_affinity + novelty_premium
    agent_gain = np.clip(agent_gain, 0.01, 1.0)

    # Apply prospect theory value function: v(x) = x^α
    v_gain = np.power(agent_gain, ALPHA)

    # ── LOSS SIDE — per-agent cost perception ──

    if price > 0:
        # ── PAID PRODUCT ──
        # Price pain is relative to what the agent can afford
        discretionary_income = income_a * 0.3  # ~30% of gross is discretionary
        discretionary_income = np.maximum(discretionary_income, 500.0)

        # Annual cost as fraction of discretionary income
        annual_cost = price * 12.0  # assume monthly subscription
        price_fraction = annual_cost / discretionary_income
        price_fraction = np.clip(price_fraction, 0.0, 1.0)

        # Prospect theory loss: v(x) = λ × |x|^β
        v_loss = la_a * np.power(price_fraction, BETA_PT)

        # Price sensitivity amplifier (some agents are more price-conscious
        # regardless of actual affordability)
        v_loss *= (0.5 + 0.5 * price_sens_a)

    else:
        # ── FREE PRODUCT ──
        # Loss is effort-based: learning curve, data migration, cognitive
        # load of evaluating and switching. Even free products have costs.
        switching_effort = switching_cost_a * 0.3
        time_risk = (1.0 - time_to_value_a) * 0.2
        behavior_cost = requires_behavior_change * 0.2

        effort_loss = switching_effort + time_risk + behavior_cost
        effort_loss = np.maximum(effort_loss, 0.01)

        # Apply loss aversion to effort costs — people overweight even
        # non-monetary losses. × 0.3 because non-monetary losses are
        # felt less intensely than equivalent monetary ones.
        v_loss = la_a * np.power(effort_loss, BETA_PT) * 0.3

        # Openness reduces effort aversion — open people don't mind
        # the hassle of trying new things
        v_loss *= (0.3 + 0.7 * (1.0 - open_a))

    prospect = v_gain - v_loss
    forces["prospect_value"][aware_mask] = prospect

    # ═══════════════════════════════════════════════════════════════
    # FORCE 2: REFERENCE PRICE ANCHORING (Tversky & Kahneman 1974)
    # ref = awareness-weighted avg of known competitor prices
    # anchor_effect = (ref - price) / max(ref, 1) × 0.3
    # ═══════════════════════════════════════════════════════════════
    # Per-agent reference price: modulated by their competitor awareness
    # Innovators know more competitors → their ref reflects full market
    # Laggards know fewer → their ref is skewed toward dominant player
    # ref_price_a may be per-agent array or scalar
    ref_has_value = np.any(ref_price_a > 0) if has_agent_params else ref_price_a > 0
    if ref_has_value and price > 0:
        # Scale reference price by competitor awareness
        # Less aware agents anchor more to the dominant/cheapest option
        agent_ref = ref_price_a * comp_frac_a + price * (1.0 - comp_frac_a)
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
    switching_friction = 0.3 + 0.7 * switching_cost_a

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
        * np.log1p(product_adoption * social_visibility_a)
        * (1.0 - benefit_certainty_a)
    )
    forces["social_proof"][aware_mask] = social

    # ═══════════════════════════════════════════════════════════════
    # FORCE 5: ANTICIPATED REGRET / FOMO (Loomes & Sugden 1982)
    # fomo_intensity × social_visibility × category_growth × social_proof_need
    # Distinct from loss aversion — this is opportunity-miss pain.
    # ═══════════════════════════════════════════════════════════════
    fomo = fomo_intensity * social_visibility_a * category_growth * fomo_a
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
        time_to_value_a * (1.0 - agent_beta) * perceived_benefit_a * DISCOUNT_SCALE
    )
    forces["hyperbolic_discounting"][aware_mask] = discount

    # ═══════════════════════════════════════════════════════════════
    # FORCE 7: IDENTITY SIGNALING (Veblen 1899, Berger & Heath 2007)
    # openness × identity_signal × social_visibility × 0.2
    # ═══════════════════════════════════════════════════════════════
    identity = open_a * identity_signal * social_visibility_a * IDENTITY_SCALE
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
