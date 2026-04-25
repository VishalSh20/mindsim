"""Wave 7 — Interventions v2.

Five mechanism-explicit interventions, each carrying cost and timeline
so ranking is by cost-per-adoption-lift rather than raw lift.

Mechanisms:
  - free_trial         : a `considering → trialing` phase edge driven by
                         the Wave 7 state-machine wiring (not a price=0
                         hack). Trials carry sunk-cost + Cialdini
                         consistency bonuses, so on delayed-benefit
                         products they produce POSITIVE lift — the
                         headline regression-fix from Waves 1-6.
  - price_cut          : reduces `price` by a fixed fraction.
  - annual_discount    : raises `present_bias_beta` toward consumption-
                         like discounting (annual commitment = lower
                         per-month perceived cost).
  - social_proof_push  : doubles `product_adoption_rate` (proxy for
                         marketing-driven WOM signal). Stronger when
                         benefit_certainty is low.
  - freemium           : drops the reference price (free option in
                         market) AND raises benefit_certainty (zero-risk
                         entry).

Each intervention is evaluated by cloning the agent array and the
config, applying the mechanism-specific mutation, restamping any
per-agent fields whose source param changed, then running
`timeline_rounds` more sub-rounds via `pipeline.simulate.run_rounds`.

The previous v1 single-shot force evaluation is gone: free-trial in
particular needs the multi-round phase machinery to exhibit its
positive lift.

Cost ranges are deliberate point estimates with a band — the design
note in WAVES-4-TO-8.md is that LLM-emitted point cost numbers are
noisy enough to be misleading; surfacing a range is more honest.
Numbers below are order-of-magnitude defaults appropriate for a
$20/mo SaaS at 1k-10k users; the LLM (or user override) can reshape
them per product.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from mindsim.engine.population import (
    PARAM_TO_AGENT_FIELD,
    REF_PRICE_AGENT_FIELD,
    restamp_agent_param,
)
from mindsim.models.config import SimulationConfig
from mindsim.models.results import InterventionResult

logger = logging.getLogger(__name__)


# Lower bound on the divisor used to compute cost_per_adoption_pp so a
# 0.1pp lift doesn't dominate ranking by virtue of being tiny.
_LIFT_FLOOR_PP = 0.5


@dataclass
class InterventionSpec:
    """Wave 7 intervention definition.

    `apply` mutates a SimulationConfig in place and returns the set of
    SimulationParams field names that were touched. The caller uses that
    set to know which per-agent fields to restamp before re-running
    rounds.
    """

    name: str
    description: str
    mechanism: str
    mechanism_type: str
    cost_usd_low: float
    cost_usd_high: float
    timeline_rounds: int
    apply: Callable[[SimulationConfig], set[str]]
    extra: dict = field(default_factory=dict)


# ─────────────────────────── mechanism handlers ───────────────────────────


def _apply_free_trial(config: SimulationConfig, *, duration_rounds: int = 3) -> set[str]:
    config.simulation_params.free_trial_active = True
    config.simulation_params.free_trial_duration_rounds = duration_rounds
    # No per-agent restamps required — the phase machinery handles routing.
    return set()


def _apply_price_cut(config: SimulationConfig, *, pct: float = 0.20) -> set[str]:
    config.simulation_params.price *= (1.0 - pct)
    # `price` is a scalar on params; not in PARAM_TO_AGENT_FIELD, so no
    # restamp needed. The forces module reads it directly.
    return set()


def _apply_annual_discount(config: SimulationConfig, *, beta_lift: float = 0.10) -> set[str]:
    cparam = config.simulation_params.present_bias_beta
    cparam.value = float(min(1.0, cparam.value + beta_lift))
    return {"present_bias_beta"}


def _apply_social_proof_push(config: SimulationConfig, *, multiplier: float = 2.0) -> set[str]:
    cparam = config.simulation_params.product_adoption_rate
    cparam.value = float(min(1.0, cparam.value * multiplier))
    return {"product_adoption_rate"}


def _apply_freemium(
    config: SimulationConfig,
    *,
    ref_price_factor: float = 0.5,
    benefit_certainty_lift: float = 0.10,
) -> set[str]:
    config.simulation_params.reference_price.value *= ref_price_factor
    cparam = config.simulation_params.benefit_certainty
    cparam.value = float(min(1.0, cparam.value + benefit_certainty_lift))
    return {"reference_price", "benefit_certainty"}


# ─────────────────────────── default catalogue ───────────────────────────


def default_interventions() -> list[InterventionSpec]:
    """The canonical 5-intervention catalogue.

    Returned as a fresh list each call so callers can mutate without
    aliasing across runs.
    """
    return [
        InterventionSpec(
            name="Free trial",
            description="Offer a 3-round free trial to eliminate upfront cost.",
            mechanism=(
                "Routes considering agents through TRIALING for trial_duration "
                "rounds. Sunk-cost + Cialdini consistency bonuses raise the "
                "post-trial conversion utility. Highest impact on delayed-"
                "benefit products where pre-trial certainty is low."
            ),
            mechanism_type="free_trial",
            cost_usd_low=5_000,
            cost_usd_high=25_000,
            timeline_rounds=4,  # trial duration + 1 conversion round
            apply=_apply_free_trial,
            extra={"trial_duration_rounds": 3},
        ),
        InterventionSpec(
            name="Price cut 20%",
            description="Reduce price by 20%.",
            mechanism=(
                "Direct loss-component reduction. Most impactful for price-"
                "sensitive segments and when the reference price is at or "
                "below the current price."
            ),
            mechanism_type="price_cut",
            cost_usd_low=50_000,
            cost_usd_high=200_000,  # revenue lost over the timeline
            timeline_rounds=3,
            apply=_apply_price_cut,
            extra={"pct": 0.20},
        ),
        InterventionSpec(
            name="Annual discount",
            description="Annual billing with ~30% savings — front-loads commitment.",
            mechanism=(
                "Annual commitment reduces per-month perceived cost; raises "
                "present_bias_beta toward consumption-like discounting. Free "
                "to ship — affects existing pricing structure only."
            ),
            mechanism_type="annual_discount",
            cost_usd_low=0,
            cost_usd_high=0,
            timeline_rounds=2,
            apply=_apply_annual_discount,
            extra={"beta_lift": 0.10},
        ),
        InterventionSpec(
            name="Social proof push",
            description="Boost social proof via testimonials, usage stats, endorsements.",
            mechanism=(
                "Doubles product_adoption_rate — proxy for marketing-driven "
                "WOM signal. Strongest when benefit_certainty is low; near-"
                "zero impact when the product is already well-known."
            ),
            mechanism_type="social_proof_push",
            cost_usd_low=10_000,
            cost_usd_high=50_000,
            timeline_rounds=3,
            apply=_apply_social_proof_push,
            extra={"multiplier": 2.0},
        ),
        InterventionSpec(
            name="Freemium tier",
            description="Add a free tier to anchor reference price down.",
            mechanism=(
                "Drops reference price (free option in market) and raises "
                "benefit_certainty (zero-risk entry). Trades long-term "
                "monetisation for adoption breadth."
            ),
            mechanism_type="freemium",
            cost_usd_low=20_000,
            cost_usd_high=100_000,
            timeline_rounds=4,
            apply=_apply_freemium,
            extra={"ref_price_factor": 0.5, "benefit_certainty_lift": 0.10},
        ),
    ]


# Kept around as a compatibility alias for any test or doc that imported
# the old name. Prefer `default_interventions()` going forward.
def STANDARD_INTERVENTIONS() -> list[InterventionSpec]:  # noqa: N802
    return default_interventions()


# ─────────────────────────── runner / ranker ───────────────────────────


def _restamp_changed(
    agents: np.ndarray,
    changed: set[str],
    config: SimulationConfig,
    rng: np.random.Generator,
) -> None:
    """Apply restamping for any param the intervention touched."""
    for name in changed:
        if name == "reference_price":
            restamp_agent_param(
                agents, "reference_price",
                config.simulation_params.reference_price, rng,
            )
            continue
        if name in PARAM_TO_AGENT_FIELD:
            cparam = getattr(config.simulation_params, name, None)
            if cparam is not None:
                restamp_agent_param(agents, name, cparam, rng)


def _simulate_intervention(
    spec: InterventionSpec,
    agents_template: np.ndarray,
    config: SimulationConfig,
    base_adoption: float,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray, SimulationConfig]:
    """Run one intervention end-to-end on a clone of the agent array.

    Returns (new_total_adoption, lift_pp, mutated_agents, mutated_config)
    so callers can chain (e.g. for the pairwise combination test).
    """
    # Avoid circular import — Wave 6 simulate -> intervention orchestration
    # happens through analyze.py; analyze imports interventions, so we
    # import simulate lazily.
    from mindsim.pipeline.simulate import run_rounds

    agents_clone = agents_template.copy()
    config_clone = config.model_copy(deep=True)

    changed = spec.apply(config_clone)
    _restamp_changed(agents_clone, changed, config_clone, rng)

    run_rounds(
        agents=agents_clone,
        params=config_clone.simulation_params,
        n_rounds=max(1, spec.timeline_rounds),
        rng=rng,
        starting_round=1,
    )

    # Adoption count: ADOPTED + LOCKED_IN over total agents.
    from mindsim.models.state import ADOPTED_PHASES
    phase_arr = agents_clone["phase"]
    decided = np.isin(
        phase_arr, np.array(ADOPTED_PHASES, dtype=phase_arr.dtype)
    ).sum()
    n_total = len(agents_clone)
    new_adoption = float(decided / n_total) if n_total > 0 else 0.0
    lift_pp = (new_adoption - base_adoption) * 100.0

    return new_adoption, lift_pp, agents_clone, config_clone


def _cost_per_pp(spec: InterventionSpec, lift_pp: float) -> float:
    """USD per percentage-point of adoption lift, midpoint costs."""
    if lift_pp <= 0:
        return math.inf
    midpoint = 0.5 * (spec.cost_usd_low + spec.cost_usd_high)
    if midpoint <= 0:
        # Free intervention — best possible cost-per-lift. Use a tiny
        # positive number so ranking still orders by lift descending
        # within the free-tier subset.
        return 0.0
    divisor = max(lift_pp, _LIFT_FLOOR_PP)
    return midpoint / divisor


def _build_result(
    spec: InterventionSpec,
    base_adoption: float,
    new_adoption: float,
    lift_pp: float,
) -> InterventionResult:
    return InterventionResult(
        name=spec.name,
        description=spec.description,
        adoption_before=base_adoption,
        adoption_after=new_adoption,
        lift_pp=lift_pp,
        mechanism=spec.mechanism,
        mechanism_type=spec.mechanism_type,
        cost_usd_low=spec.cost_usd_low,
        cost_usd_high=spec.cost_usd_high,
        timeline_rounds=spec.timeline_rounds,
        cost_per_adoption_pp=_cost_per_pp(spec, lift_pp),
    )


def _classify_additivity(lift_a: float, lift_b: float, lift_ab: float, eps: float = 0.5) -> str:
    """Sub/super/additive classification with a percentage-point tolerance."""
    expected = lift_a + lift_b
    if lift_ab > expected + eps:
        return "super_additive"
    if lift_ab < expected - eps:
        return "sub_additive"
    return "additive"


def rank_interventions(
    agents: np.ndarray,
    config: SimulationConfig,
    base_adoption: float,
    interventions: list[InterventionSpec] | None = None,
    rng: np.random.Generator | None = None,
    pairwise_combo: bool = True,
) -> list[InterventionResult]:
    """Evaluate, rank, and (optionally) pairwise-combine interventions.

    Ranking key — primary: `cost_per_adoption_pp` ascending; secondary:
    `lift_pp` descending; tertiary: `timeline_rounds` ascending.

    When `pairwise_combo=True`, the top-2 *positive-lift* interventions
    by cost-per-pp are also evaluated as a combined run (A applied then
    B applied to the same cloned agents). The combo is appended to the
    result list with `combined_with` set and `additivity` classified.

    Args:
        agents: post-baseline agent array (typically `sim_result._agents`).
        config: SimulationConfig that produced `base_adoption`.
        base_adoption: total_adoption rate before any intervention.
        interventions: candidates. Defaults to `default_interventions()`.
        rng: shared RNG. A child generator is forked per evaluation so the
            order of evaluation doesn't affect any single intervention's
            outcome.
        pairwise_combo: whether to add the top-2 combination row.

    Returns:
        A list of InterventionResult, sorted by the ranking key. The
        combo row (if any) is the LAST element regardless of rank — it's
        diagnostic, not a candidate to ship on its own.
    """
    if interventions is None:
        interventions = default_interventions()
    if rng is None:
        rng = np.random.default_rng(42)

    results: list[InterventionResult] = []
    sims_by_name: dict[str, tuple[float, float, np.ndarray, SimulationConfig]] = {}

    for i, spec in enumerate(interventions):
        # Fork a per-spec child RNG so evaluation order doesn't matter.
        child_rng = np.random.default_rng(rng.integers(0, 2**32 - 1) + i)
        new_adoption, lift_pp, agents_after, config_after = _simulate_intervention(
            spec, agents, config, base_adoption, child_rng
        )
        sims_by_name[spec.name] = (new_adoption, lift_pp, agents_after, config_after)
        results.append(_build_result(spec, base_adoption, new_adoption, lift_pp))
        logger.debug(
            "intervention %s: %.1f%% (lift %+.2fpp, $%s/pp)",
            spec.name, new_adoption * 100, lift_pp,
            f"{results[-1].cost_per_adoption_pp:.0f}"
            if math.isfinite(results[-1].cost_per_adoption_pp)
            else "inf",
        )

    # Primary: cost_per_adoption_pp ascending. Within ties, lift descending.
    # Negative-lift interventions get sorted to the end via cost==inf.
    results.sort(
        key=lambda r: (r.cost_per_adoption_pp, -r.lift_pp, r.timeline_rounds)
    )

    if pairwise_combo:
        positive = [r for r in results if r.lift_pp > 0]
        if len(positive) >= 2:
            top_a, top_b = positive[0], positive[1]
            spec_a = next(s for s in interventions if s.name == top_a.name)
            spec_b = next(s for s in interventions if s.name == top_b.name)

            # Apply A then B on a single shared clone.
            from mindsim.pipeline.simulate import run_rounds
            agents_clone = agents.copy()
            config_clone = config.model_copy(deep=True)
            child_rng = np.random.default_rng(rng.integers(0, 2**32 - 1) + 999)

            changed = set()
            changed |= spec_a.apply(config_clone)
            changed |= spec_b.apply(config_clone)
            _restamp_changed(agents_clone, changed, config_clone, child_rng)
            run_rounds(
                agents=agents_clone,
                params=config_clone.simulation_params,
                n_rounds=max(spec_a.timeline_rounds, spec_b.timeline_rounds),
                rng=child_rng,
                starting_round=1,
            )

            from mindsim.models.state import ADOPTED_PHASES
            phase_arr = agents_clone["phase"]
            decided = np.isin(
                phase_arr, np.array(ADOPTED_PHASES, dtype=phase_arr.dtype)
            ).sum()
            n_total = len(agents_clone)
            combo_adoption = float(decided / n_total) if n_total > 0 else 0.0
            combo_lift = (combo_adoption - base_adoption) * 100.0

            combo_cost_low = spec_a.cost_usd_low + spec_b.cost_usd_low
            combo_cost_high = spec_a.cost_usd_high + spec_b.cost_usd_high
            combo_midpoint = 0.5 * (combo_cost_low + combo_cost_high)
            combo_cpp = (
                combo_midpoint / max(combo_lift, _LIFT_FLOOR_PP)
                if combo_lift > 0
                else math.inf
            )
            additivity = _classify_additivity(top_a.lift_pp, top_b.lift_pp, combo_lift)

            results.append(InterventionResult(
                name=f"{top_a.name} + {top_b.name}",
                description=(
                    f"Combined application of '{top_a.name}' and "
                    f"'{top_b.name}' (top-2 by cost-per-lift)."
                ),
                adoption_before=base_adoption,
                adoption_after=combo_adoption,
                lift_pp=combo_lift,
                mechanism=(
                    f"{additivity} combination — "
                    f"single-intervention lifts {top_a.lift_pp:+.2f}pp + "
                    f"{top_b.lift_pp:+.2f}pp = {top_a.lift_pp + top_b.lift_pp:+.2f}pp "
                    f"vs combined {combo_lift:+.2f}pp."
                ),
                mechanism_type="combo",
                cost_usd_low=combo_cost_low,
                cost_usd_high=combo_cost_high,
                timeline_rounds=max(spec_a.timeline_rounds, spec_b.timeline_rounds),
                cost_per_adoption_pp=combo_cpp,
                combined_with=top_b.name,
                additivity=additivity,
            ))

    return results
