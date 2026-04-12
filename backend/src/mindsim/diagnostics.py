"""Diagnostic dump — writes detailed pipeline state to a JSON file.

When --dump is passed, each pipeline stage appends its output to a
diagnostic file so you can inspect:
  - Research results (verified competitors, confidence flags, trends)
  - Calibrated parameters (every CalibratedParam with basis/confidence)
  - Per-agent data (archetype, traits, forces, probability, decision)
  - Sensitivity analysis and intervention results

Usage: mindsim --dump "product description"
Output: mindsim_dump.json (or path specified by --dump-path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class DiagnosticDump:
    """Accumulates pipeline stage outputs and writes them to a file."""

    def __init__(self, path: str = "mindsim_dump.json"):
        self.path = Path(path)
        self.data: dict = {
            "stages": {},
            "warnings": [],
        }

    def record_understand(self, profile) -> None:
        """Record Stage 1: Understand output."""
        stage_data = {
            "product_name": profile.name,
            "price": profile.price,
            "price_model": profile.price_model,
            "billing_period": profile.billing_period,
            "category": profile.category,
            "target_audience": profile.target_audience,
            "target_audience_inferred": profile.target_audience_inferred,
            "value_proposition": profile.value_proposition,
            "missing_information": profile.missing_information,
            "competitors": [
                {"name": c.name, "price": c.price, "price_model": c.price_model}
                for c in profile.competitors
            ],
            "research_plan": {
                "priority_queries": [
                    {
                        "query": q.query,
                        "goal": q.goal,
                        "confidence_threshold": q.confidence_threshold,
                        "fallback_assumption": q.fallback_assumption,
                    }
                    for q in profile.research_plan.priority_queries
                ],
                "optional_queries": [
                    {
                        "query": q.query,
                        "goal": q.goal,
                        "trigger_condition": q.trigger_condition,
                    }
                    for q in profile.research_plan.optional_queries
                ],
            },
        }

        # Include validation results if available
        validation = getattr(profile, "_validation", None)
        if validation is not None:
            stage_data["validation"] = {
                "has_critical_errors": validation.get("has_critical_errors", False),
                "corrections_applied": [
                    {
                        "field": v.get("field"),
                        "extracted_value": v.get("extracted_value"),
                        "correct_value": v.get("correct_value"),
                        "reason": v.get("reason"),
                    }
                    for v in validation.get("validations", [])
                    if v.get("status") == "WRONG"
                ],
            }

        self.data["stages"]["1_understand"] = stage_data

    def record_research(self, market) -> None:
        """Record Stage 2: Research output."""
        self.data["stages"]["2_research"] = {
            "research_skipped": market.research_skipped,
            "research_cost_credits": market.research_cost,
            "verified_competitors": [
                {
                    "name": c.name,
                    "confirmed_price": c.confirmed_price,
                    "price_model": c.price_model,
                    "has_free_tier": c.has_free_tier,
                    "source_url": c.source_url,
                }
                for c in market.verified_competitors
            ],
            "discovered_competitors": [
                {
                    "name": c.name,
                    "confirmed_price": c.confirmed_price,
                    "has_free_tier": c.has_free_tier,
                    "source_url": c.source_url,
                }
                for c in market.discovered_competitors
            ],
            "category_penetration": market.category_penetration,
            "category_penetration_confidence": market.category_penetration_confidence,
            "category_growth": market.category_growth,
            "category_maturity": market.category_maturity,
            "low_confidence_flags": market.low_confidence_flags,
        }

    def record_calibration(self, config) -> None:
        """Record Stage 3: Calibrate output — all parameters with provenance."""
        params = config.simulation_params

        def _cparam_dict(cp) -> dict:
            d = {
                "value": cp.value,
                "basis": cp.basis,
                "confidence": cp.confidence,
            }
            if cp.by_archetype:
                d["by_archetype"] = cp.by_archetype
            return d

        self.data["stages"]["3_calibrate"] = {
            "price": params.price,
            "reference_price": {
                "value": params.reference_price.value,
                "confidence": params.reference_price.confidence,
                "basis": params.reference_price.basis,
                "components": [
                    {"source": c.source, "price": c.price, "weight": c.weight}
                    for c in params.reference_price.components
                ],
                **({"by_archetype": params.reference_price.by_archetype} if params.reference_price.by_archetype else {}),
            },
            "parameters": {
                "category_penetration": _cparam_dict(params.category_penetration),
                "category_growth": _cparam_dict(params.category_growth),
                "benefit_certainty": _cparam_dict(params.benefit_certainty),
                "perceived_benefit": _cparam_dict(params.perceived_benefit),
                "time_to_value": _cparam_dict(params.time_to_value),
                "requires_behavior_change": _cparam_dict(params.requires_behavior_change),
                "switching_cost": _cparam_dict(params.switching_cost),
                "social_visibility": _cparam_dict(params.social_visibility),
                "identity_signal": _cparam_dict(params.identity_signal),
                "present_bias_beta": _cparam_dict(params.present_bias_beta),
                "fomo_intensity": _cparam_dict(params.fomo_intensity),
                "product_adoption_rate": _cparam_dict(params.product_adoption_rate),
            },
            "awareness_by_archetype": {
                "innovator": params.awareness.innovator,
                "early_adopter": params.awareness.early_adopter,
                "early_majority": params.awareness.early_majority,
                "late_majority": params.awareness.late_majority,
                "laggard": params.awareness.laggard,
            },
            "population_config": {
                "income_mean_log": config.population_config.income_mean_log,
                "income_sigma": config.population_config.income_sigma,
                "market_segment": config.population_config.market_segment,
            },
            "assumptions": [
                {
                    "id": a.id,
                    "parameter": a.parameter,
                    "value": a.value,
                    "basis": a.basis,
                    "confidence": a.confidence,
                    "sensitivity": a.sensitivity,
                }
                for a in config.assumptions
            ],
        }

    def record_simulation(self, sim_result, config) -> None:
        """Record Stage 4: Simulate — per-agent data + aggregate results."""
        agents = sim_result._agents
        forces = sim_result._agent_forces
        probs = sim_result._agent_probs
        decisions = sim_result._agent_decisions

        archetype_names = [
            "innovator", "early_adopter", "early_majority",
            "late_majority", "laggard",
        ]

        # Per-agent dump (all 1000 agents)
        agent_rows = []
        if agents is not None:
            for i in range(len(agents)):
                row = {
                    "id": i,
                    "archetype": archetype_names[int(agents[i]["archetype_id"])],
                    "aware": bool(agents[i]["aware"]),
                    "income": round(float(agents[i]["income"]), 2),
                    "loss_aversion_lambda": round(float(agents[i]["loss_aversion_lambda"]), 4),
                    "status_quo_bias": round(float(agents[i]["status_quo_bias"]), 4),
                    "social_proof_need": round(float(agents[i]["social_proof_need"]), 4),
                    "fomo_susceptibility": round(float(agents[i]["fomo_susceptibility"]), 4),
                    "openness": round(float(agents[i]["openness"]), 4),
                    "neuroticism": round(float(agents[i]["neuroticism"]), 4),
                    "agreeableness": round(float(agents[i]["agreeableness"]), 4),
                    "conscientiousness": round(float(agents[i]["conscientiousness"]), 4),
                    "novelty_weight": round(float(agents[i]["novelty_weight"]), 4),
                    "price_sensitivity": round(float(agents[i]["price_sensitivity"]), 4),
                    "competitor_awareness_frac": round(float(agents[i]["competitor_awareness_frac"]), 4),
                    "agent_perceived_benefit": round(float(agents[i]["agent_perceived_benefit"]), 4),
                    "agent_benefit_certainty": round(float(agents[i]["agent_benefit_certainty"]), 4),
                    "agent_switching_cost": round(float(agents[i]["agent_switching_cost"]), 4),
                    "agent_reference_price": round(float(agents[i]["agent_reference_price"]), 2),
                    "agent_social_visibility": round(float(agents[i]["agent_social_visibility"]), 4),
                    "agent_time_to_value": round(float(agents[i]["agent_time_to_value"]), 4),
                }

                # Per-agent forces
                if forces is not None and agents[i]["aware"]:
                    row["forces"] = {}
                    for fname, farr in forces.items():
                        val = farr[i]
                        row["forces"][fname] = round(float(val), 6) if not np.isnan(val) else None

                    # Total utility
                    total = sum(
                        v for v in row["forces"].values() if v is not None
                    )
                    row["total_utility"] = round(total, 6)

                if probs is not None:
                    row["adoption_probability"] = round(float(probs[i]), 6)
                if decisions is not None:
                    row["adopted"] = bool(decisions[i])

                agent_rows.append(row)

        # Aggregate stats
        sim_data = {
            "total_adoption": sim_result.total_adoption,
            "aware_adoption": sim_result.aware_adoption,
            "n_agents": sim_result.n_agents,
            "n_aware": sim_result.n_aware,
            "force_decomposition": sim_result.force_decomposition.as_dict(),
            "by_archetype": [
                {
                    "name": s.name,
                    "adoption_rate": s.adoption_rate,
                    "count": s.count,
                    "total": s.total,
                }
                for s in sim_result.by_archetype
            ],
            "by_income": [
                {
                    "bracket": s.bracket,
                    "adoption_rate": s.adoption_rate,
                    "count": s.count,
                    "total": s.total,
                }
                for s in sim_result.by_income
            ],
            "by_proximity": {
                "locked": sim_result.by_proximity.locked,
                "convertible": sim_result.by_proximity.convertible,
                "unreachable": sim_result.by_proximity.unreachable,
            },
            "agents": agent_rows,
        }

        self.data["stages"]["4_simulate"] = sim_data

    def record_analysis(self, report) -> None:
        """Record Stage 5: Analyze — sensitivity, confidence, interventions."""
        self.data["stages"]["5_analyze"] = {
            "confidence_band": {
                "central": report.confidence_band.central,
                "low": report.confidence_band.low,
                "high": report.confidence_band.high,
                "drivers": report.confidence_band.drivers,
            },
            "sensitivity": [
                {
                    "parameter": s.parameter,
                    "base_adoption": s.base_adoption,
                    "low_adoption": s.low_adoption,
                    "high_adoption": s.high_adoption,
                    "swing_pp": s.swing,
                    "confidence": s.confidence,
                }
                for s in report.sensitivity
            ],
            "interventions": [
                {
                    "name": i.name,
                    "adoption_before": i.adoption_before,
                    "adoption_after": i.adoption_after,
                    "lift_pp": i.lift_pp,
                    "mechanism": i.mechanism,
                }
                for i in report.interventions
            ],
        }

    def record_event(self, event_text: str, event_result) -> None:
        """Record an event processing result."""
        events = self.data.setdefault("events", [])
        events.append({
            "event_text": event_text,
            "adoption_before": event_result.adoption_before,
            "adoption_after": event_result.adoption_after,
            "adoption_delta": event_result.adoption_delta,
            "force_adjustments": [
                {
                    "force": a.force,
                    "magnitude": a.magnitude,
                    "mechanism": a.mechanism,
                    "new_value": a.new_value,
                }
                for a in event_result.force_adjustments
            ],
            "segment_effects": event_result.segment_effects,
            "second_order_effects": event_result.second_order_effects,
        })

    def add_warning(self, warning: str) -> None:
        """Add a diagnostic warning (discrepancy, suspicious value, etc.)."""
        self.data["warnings"].append(warning)

    def write(self) -> None:
        """Write the accumulated diagnostic data to the JSON file."""
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        logger.info(f"Diagnostic dump written to {self.path}")
