"""Population generator — archetype-based agent generation via NumPy.

Generates N agents as a structured NumPy array. Each agent has:
- Archetype assignment (by Rogers share weights)
- Behavioral parameters (sampled from archetype distributions)
- Loss aversion with Gächter mixture model
- Income from lognormal distribution
- Personality-economics correlations via Cholesky decomposition
"""

from __future__ import annotations

import logging

import numpy as np

from mindsim.engine.archetypes import ArchetypeSet, load_archetypes
from mindsim.engine.feature_weights import (
    FEATURE_CATEGORIES,
    FeatureWeights,
    load_feature_weights,
)
from mindsim.models.config import ARCHETYPE_NAMES, PopulationConfig, SimulationParams, resolve_archetype_value
from mindsim.models.state import Phase

logger = logging.getLogger(__name__)


# Structured dtype for agent arrays
# v2-middle Wave 2 additions: feature_weight_* fields (one per category)
# stamped from config/feature_weights.yaml with per-agent jitter.
AGENT_DTYPE = np.dtype([
    ("archetype_id", np.int8),
    ("loss_aversion_lambda", np.float32),
    ("status_quo_bias", np.float32),
    ("social_proof_need", np.float32),
    ("novelty_weight", np.float32),
    ("price_sensitivity", np.float32),
    ("fomo_susceptibility", np.float32),
    ("openness", np.float32),
    ("neuroticism", np.float32),
    ("agreeableness", np.float32),
    ("conscientiousness", np.float32),
    ("income", np.float32),
    ("aware", np.bool_),
    ("competitor_awareness_frac", np.float32),
    # Per-agent product perception fields (stamped from archetype-specific calibration)
    ("agent_perceived_benefit", np.float32),
    ("agent_benefit_certainty", np.float32),
    ("agent_switching_cost", np.float32),
    ("agent_reference_price", np.float32),
    ("agent_social_visibility", np.float32),
    ("agent_time_to_value", np.float32),
    # v2-middle Wave 2: per-agent feature-category weights (sum to 1.0 per agent)
    ("feature_weight_core_value", np.float32),
    ("feature_weight_social_signal", np.float32),
    ("feature_weight_ongoing_cost", np.float32),
    ("feature_weight_switching_friction_reducer", np.float32),
    # v2-middle Wave 3: agent state + memory.
    # `phase` is the authoritative state (Phase enum int8). `aware` above
    # mirrors `phase != UNAWARE` for back-compat with force-slicing code.
    ("phase", np.int8),
    ("awareness_strength", np.float32),   # 0-1, graded; decays per round
    ("tenure_current_solution", np.float32),  # months with incumbent
    ("investment_depth", np.float32),      # 0-1, grows while ADOPTED
    ("trial_outcome", np.int8),            # -1 none, 0 negative, 1 positive
    ("trial_rounds_remaining", np.int8),   # 0 when not TRIALING
    ("cluster_id", np.uint8),              # placeholder — W4 fills it
])


def generate_population(
    n: int = 1000,
    population_config: PopulationConfig | None = None,
    archetype_set: ArchetypeSet | None = None,
    sim_params: SimulationParams | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a population of agents.

    Args:
        n: Number of agents to generate.
        population_config: Income distribution and market segment config.
        archetype_set: Archetype definitions. Loads from YAML if None.
        rng: Random number generator for reproducibility.

    Returns:
        Structured NumPy array of shape (n,) with AGENT_DTYPE fields.
    """
    if population_config is None:
        population_config = PopulationConfig()
    if archetype_set is None:
        archetype_set = load_archetypes()
    if rng is None:
        rng = np.random.default_rng()

    agents = np.zeros(n, dtype=AGENT_DTYPE)
    archetype_names = archetype_set.names
    shares = np.array(archetype_set.shares)

    # Normalize shares (should already sum to ~1.0)
    shares = shares / shares.sum()

    # --- Assign archetypes by share weights ---
    archetype_ids = rng.choice(
        len(archetype_names), size=n, p=shares
    ).astype(np.int8)
    agents["archetype_id"] = archetype_ids

    # --- Sample personality traits (Big Five subset) ---
    # These are sampled independently first, then correlations applied
    # Norms from Costa & McCrae 1992 NEO-PI-R, scaled to [0, 1]
    neuroticism = rng.normal(0.5, 0.15, size=n).astype(np.float32)
    openness_raw = rng.normal(0.5, 0.15, size=n).astype(np.float32)
    agreeableness_raw = rng.normal(0.5, 0.15, size=n).astype(np.float32)
    conscientiousness_raw = rng.normal(0.5, 0.15, size=n).astype(np.float32)

    # Clip to [0, 1]
    neuroticism = np.clip(neuroticism, 0.0, 1.0)
    openness_raw = np.clip(openness_raw, 0.0, 1.0)
    agreeableness_raw = np.clip(agreeableness_raw, 0.0, 1.0)
    conscientiousness_raw = np.clip(conscientiousness_raw, 0.0, 1.0)

    agents["neuroticism"] = neuroticism
    agents["agreeableness"] = agreeableness_raw
    agents["conscientiousness"] = conscientiousness_raw

    # --- Sample behavioral parameters per archetype ---
    corr = archetype_set.correlations
    mixture = archetype_set.loss_aversion_mixture

    for i, aname in enumerate(archetype_names):
        mask = archetype_ids == i
        count = mask.sum()
        if count == 0:
            continue

        aconf = archetype_set.archetypes[aname]

        # Sample each distribution
        for param_name, pdist in aconf.distributions.items():
            if param_name == "loss_aversion_lambda":
                # Handled separately with mixture model
                continue
            if param_name == "openness":
                # Will blend with personality trait
                archetype_openness = rng.normal(pdist.mean, pdist.std, size=count)
                archetype_openness = np.clip(archetype_openness, pdist.min, pdist.max)
                # Blend: 60% archetype, 40% personality
                blended = 0.6 * archetype_openness + 0.4 * openness_raw[mask]
                agents["openness"][mask] = np.clip(
                    blended, pdist.min, pdist.max
                ).astype(np.float32)
                continue

            values = rng.normal(pdist.mean, pdist.std, size=count)
            values = np.clip(values, pdist.min, pdist.max)
            agents[param_name][mask] = values.astype(np.float32)

        # --- Loss aversion with mixture model (Gächter et al. 2022) ---
        la_dist = aconf.distributions["loss_aversion_lambda"]

        # Determine which agents get near-zero loss aversion
        is_near_zero = rng.random(count) < mixture.near_zero_weight

        # Sample from archetype distribution
        la_archetype = rng.normal(la_dist.mean, la_dist.std, size=count)
        # Sample from near-zero distribution
        la_near_zero = rng.normal(mixture.near_zero_mean, mixture.near_zero_std, size=count)

        # Mix
        la_values = np.where(is_near_zero, la_near_zero, la_archetype)
        la_values = np.clip(la_values, la_dist.min, la_dist.max)

        # Apply neuroticism correlation (Lauriola & Levin 2001, r≈0.3)
        neuroticism_effect = corr.neuroticism_loss_aversion * (
            neuroticism[mask] - 0.5
        ) * la_dist.std
        la_values = la_values + neuroticism_effect
        la_values = np.clip(la_values, la_dist.min, la_dist.max)

        agents["loss_aversion_lambda"][mask] = la_values.astype(np.float32)

    # --- Apply agreeableness → social_proof correlation ---
    # Higher agreeableness → higher social proof need
    social_adj = corr.agreeableness_social_proof * (
        agreeableness_raw - 0.5
    ) * 0.15  # scale factor
    agents["social_proof_need"] = np.clip(
        agents["social_proof_need"] + social_adj, 0.0, 1.0
    ).astype(np.float32)

    # --- Income from lognormal ---
    income = rng.lognormal(
        population_config.income_mean_log,
        population_config.income_sigma,
        size=n,
    ).astype(np.float32)
    agents["income"] = income

    # --- Stamp per-archetype product perception params ---
    if sim_params is not None:
        _stamp_product_params(agents, archetype_ids, archetype_names, sim_params, rng)

    # --- v2-middle Wave 2: stamp per-agent feature_weights from YAML ---
    _stamp_feature_weights(agents, archetype_ids, archetype_names, rng)

    # --- Awareness is set later by the simulation stage ---
    agents["aware"] = True  # default, overridden in simulate
    agents["competitor_awareness_frac"] = 0.5  # default

    # --- v2-middle Wave 3: initial state ---
    # Everyone starts UNAWARE. apply_awareness() promotes a subset to AWARE
    # and sets awareness_strength. Multi-round simulate (Commit 2) advances
    # agents through the rest of the phases.
    agents["phase"] = int(Phase.UNAWARE)
    agents["awareness_strength"] = 0.0
    agents["tenure_current_solution"] = 0.0   # filled more richly in Commit 2
    agents["investment_depth"] = 0.0
    agents["trial_outcome"] = -1
    agents["trial_rounds_remaining"] = 0
    agents["cluster_id"] = 0                   # single cluster until Wave 4

    return agents


def _stamp_feature_weights(
    agents: np.ndarray,
    archetype_ids: np.ndarray,
    archetype_names: list[str],
    rng: np.random.Generator,
) -> None:
    """Stamp per-agent feature_weight_* fields from feature_weights.yaml.

    Per archetype: draw the base weight vector, apply multiplicative
    lognormal-ish jitter (sigma from YAML), renormalise so each agent's
    weights sum to exactly 1.0.
    """
    try:
        fw: FeatureWeights | None = load_feature_weights()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "feature_weights YAML unreadable (%s); using uniform 0.25 weights", exc
        )
        fw = None

    sigma = fw.noise_sigma if fw is not None else 0.0

    for i, aname in enumerate(archetype_names):
        mask = archetype_ids == i
        count = int(mask.sum())
        if count == 0:
            continue

        if fw is not None:
            base = fw.weights_for(aname)
        else:
            base = {c: 0.25 for c in FEATURE_CATEGORIES}

        base_arr = np.array(
            [base[c] for c in FEATURE_CATEGORIES], dtype=np.float64
        )
        # Per-agent multiplicative noise. Shape: (count, 4).
        if sigma > 0.0:
            noise = rng.normal(loc=1.0, scale=sigma, size=(count, len(FEATURE_CATEGORIES)))
        else:
            noise = np.ones((count, len(FEATURE_CATEGORIES)), dtype=np.float64)
        jittered = np.clip(base_arr[None, :] * noise, 1e-9, None)
        jittered /= jittered.sum(axis=1, keepdims=True)

        for j, cat in enumerate(FEATURE_CATEGORIES):
            field = f"feature_weight_{cat}"
            agents[field][mask] = jittered[:, j].astype(np.float32)


# Map from SimulationParams field name to AGENT_DTYPE field name
PARAM_TO_AGENT_FIELD = {
    "perceived_benefit": "agent_perceived_benefit",
    "benefit_certainty": "agent_benefit_certainty",
    "switching_cost": "agent_switching_cost",
    "social_visibility": "agent_social_visibility",
    "time_to_value": "agent_time_to_value",
}

# reference_price is handled separately (ReferencePriceParam, not 0-1 scale)
REF_PRICE_AGENT_FIELD = "agent_reference_price"


def _stamp_product_params(
    agents: np.ndarray,
    archetype_ids: np.ndarray,
    archetype_names: list[str],
    sim_params: SimulationParams,
    rng: np.random.Generator,
) -> None:
    """Stamp per-archetype product perception values onto agents.

    For each of the 6 product params, resolves the archetype-specific value
    (or falls back to base), adds ±3.5% noise, and writes to agent array.
    """
    for param_name, agent_field in PARAM_TO_AGENT_FIELD.items():
        cparam = getattr(sim_params, param_name)
        for i, aname in enumerate(archetype_names):
            mask = archetype_ids == i
            count = int(mask.sum())
            if count == 0:
                continue
            base = resolve_archetype_value(cparam.value, cparam.by_archetype, aname)
            sigma = max(base * 0.035, 0.005)  # ±3.5% noise, min sigma for near-zero
            noisy = rng.normal(base, sigma, size=count)
            noisy = np.clip(noisy, 0.0, 1.0)
            agents[agent_field][mask] = noisy.astype(np.float32)

    # Reference price (dollar-denominated, not 0-1)
    ref = sim_params.reference_price
    for i, aname in enumerate(archetype_names):
        mask = archetype_ids == i
        count = int(mask.sum())
        if count == 0:
            continue
        base = resolve_archetype_value(ref.value, ref.by_archetype, aname)
        sigma = max(base * 0.05, 0.01)  # ±5% noise for prices
        noisy = rng.normal(base, sigma, size=count)
        noisy = np.maximum(noisy, 0.0)
        agents[REF_PRICE_AGENT_FIELD][mask] = noisy.astype(np.float32)


def restamp_agent_param(
    agents: np.ndarray,
    param_name: str,
    cparam,
    rng: np.random.Generator,
) -> None:
    """Re-stamp one per-agent product param field after a param override.

    Used by sensitivity analysis and interventions to update agent arrays
    when a CalibratedParam is modified for a rerun.
    """
    if param_name in PARAM_TO_AGENT_FIELD:
        agent_field = PARAM_TO_AGENT_FIELD[param_name]
        for i, aname in enumerate(ARCHETYPE_NAMES):
            mask = agents["archetype_id"] == i
            count = int(mask.sum())
            if count == 0:
                continue
            base = resolve_archetype_value(cparam.value, cparam.by_archetype, aname)
            sigma = max(base * 0.035, 0.005)
            noisy = rng.normal(base, sigma, size=count)
            noisy = np.clip(noisy, 0.0, 1.0)
            agents[agent_field][mask] = noisy.astype(np.float32)
    elif param_name == "reference_price":
        for i, aname in enumerate(ARCHETYPE_NAMES):
            mask = agents["archetype_id"] == i
            count = int(mask.sum())
            if count == 0:
                continue
            base = resolve_archetype_value(cparam.value, cparam.by_archetype, aname)
            sigma = max(base * 0.05, 0.01)
            noisy = rng.normal(base, sigma, size=count)
            noisy = np.maximum(noisy, 0.0)
            agents[REF_PRICE_AGENT_FIELD][mask] = noisy.astype(np.float32)


def apply_awareness(
    agents: np.ndarray,
    awareness_by_archetype: dict[str, float],
    archetype_names: list[str],
    rng: np.random.Generator | None = None,
    archetype_set: ArchetypeSet | None = None,
) -> np.ndarray:
    """Apply awareness filter based on archetype-specific probabilities.

    Modifies agents in-place and returns the same array.

    Args:
        agents: Agent array.
        awareness_by_archetype: Dict mapping archetype name to awareness probability.
        archetype_names: Ordered list of archetype names.
        rng: Random number generator.
        archetype_set: For competitor awareness fractions.

    Returns:
        Modified agents array with 'aware' and 'competitor_awareness_frac' set.
    """
    if rng is None:
        rng = np.random.default_rng()
    if archetype_set is None:
        archetype_set = load_archetypes()

    awareness_probs = np.array([
        awareness_by_archetype.get(name, 0.5) for name in archetype_names
    ])

    # Set awareness per agent based on their archetype
    for i, aname in enumerate(archetype_names):
        mask = agents["archetype_id"] == i
        count = mask.sum()
        if count == 0:
            continue

        # Product awareness
        aware_mask = rng.random(count) < awareness_probs[i]
        agents["aware"][mask] = aware_mask

        # v2-middle Wave 3: promote UNAWARE → AWARE for this subset, and
        # seed awareness_strength. Newly-aware agents get a moderate strength
        # (0.5) meaning "I know about this but haven't engaged yet"; the
        # Wave 3 Commit 2 state machine then decides whether they cross the
        # consideration threshold. Vectorised via global-index slicing.
        archetype_indices = np.flatnonzero(mask)
        becomes_aware_idx = archetype_indices[aware_mask]
        stays_unaware_idx = archetype_indices[~aware_mask]
        agents["phase"][becomes_aware_idx] = int(Phase.AWARE)
        agents["phase"][stays_unaware_idx] = int(Phase.UNAWARE)
        agents["awareness_strength"][becomes_aware_idx] = 0.5
        agents["awareness_strength"][stays_unaware_idx] = 0.0

        # Competitor awareness fraction
        comp_frac = archetype_set.archetypes[aname].awareness.competitor_fraction
        # Add some noise
        comp_fracs = rng.normal(comp_frac, 0.1, size=count)
        comp_fracs = np.clip(comp_fracs, 0.0, 1.0)
        agents["competitor_awareness_frac"][mask] = comp_fracs.astype(np.float32)

    return agents
