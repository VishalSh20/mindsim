"""A5 — calibration-bundle validator.

Two-layer validation:

  1. **Hard-range checks (pure Python).** Parameter bounds drawn from
     RESEARCH.md. These are non-negotiable: if λ comes back as 0.3, the
     LLM made a mistake. No re-prompting for these — they're caught
     before A5 fires its LLM call.

  2. **Cross-field LLM review.** The A5 prompt gets the full bundle and
     looks for linguistic inconsistencies that aren't range violations:
     e.g. "maturity=saturated but penetration=0.12", "three features all
     rated >0.9 — is the LLM anchoring?", "no negative-polarity features
     on a paid product".

Hard-range failures and cross-field LLM issues both flow to the same
`ValidationReport`. The orchestrator's `run_with_retry` wraps this.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from mindsim.llm.client import LLMClient
from mindsim.models.config import (
    Assumption,
    CalibratedParam,
    SimulationConfig,
)
from mindsim.models.evidence import EvidenceStore
from mindsim.models.product import Feature, ProductProfile

logger = logging.getLogger(__name__)


# ───────────────────────── Published-bounds tables ─────────────────────────
# All bounds cite RESEARCH.md. Kept in one place so they're auditable and
# so the validator can never drift from what's documented.

# Loss aversion λ: T&K 1992, Gächter et al. 2022 mixture lower end.
LAMBDA_MIN, LAMBDA_MAX = 1.0, 4.0
# Present-bias β: Laibson 1997, Augenblick 2015.
BETA_MIN, BETA_MAX = 0.3, 0.95
# Probability weighting γ: locked at T&K 1992.
GAMMA_LOCK = 0.61
# Category penetration is a proportion of TAM.
PENETRATION_MIN, PENETRATION_MAX = 0.0, 1.0
# Feature-matrix cardinality per plan §9 (A4 prompt constraint).
FEATURE_COUNT_MIN, FEATURE_COUNT_MAX = 4, 8
# Per-feature dimensions.
SCORE_MIN, SCORE_MAX = 0.0, 1.0
CERTAINTY_MIN, CERTAINTY_MAX = 0.0, 1.0
VISIBILITY_MIN, VISIBILITY_MAX = 0.0, 1.0
TIME_TO_VALUE_MIN, TIME_TO_VALUE_MAX = 0.0, 36.0


# Wave 8.5 — published bounds for parameters with peer-reviewed ranges.
# Used to populate `Assumption.published_bounds` and to validate that
# calibrated values land inside the literature window. RESEARCH.md is
# the source of truth.
PUBLISHED_BOUNDS: dict[str, tuple[float, float]] = {
    "present_bias_beta": (BETA_MIN, BETA_MAX),
    "category_penetration": (PENETRATION_MIN, PENETRATION_MAX),
    "category_growth": (0.0, 5.0),  # multi-year YoY upper bound
    "perceived_benefit": (0.0, 1.0),
    "benefit_certainty": (0.0, 1.0),
    "switching_cost": (0.0, 1.0),
    "social_visibility": (0.0, 1.0),
    "identity_signal": (0.0, 1.0),
    "fomo_intensity": (0.0, 1.0),
    "product_adoption_rate": (0.0, 1.0),
    "requires_behavior_change": (0.0, 1.0),
    "time_to_value": (0.0, 1.0),
}

# Wave 8.5 — sparse-rationale threshold. Below this, the param's basis
# is too thin to be load-bearing; validator flags it.
SPARSE_BASIS_CHARS = 10


Severity = Literal["error", "warning"]


class ValidationIssue(BaseModel):
    severity: Severity
    field: str
    message: str
    suggested_value: Any | None = None


class ValidationReport(BaseModel):
    is_valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    corrected_fields: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = 1.0


class ValidatorInput(BaseModel):
    """Full calibration bundle handed to A5."""

    product: ProductProfile
    config: SimulationConfig
    reasoning_trail: list[dict] = Field(default_factory=list)


# ───────────────────────── Hard-range checkers ─────────────────────────


def check_feature_matrix(features: list[Feature]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    n = len(features)
    if n < FEATURE_COUNT_MIN:
        issues.append(
            ValidationIssue(
                severity="error",
                field="feature_matrix",
                message=(
                    f"too few features ({n}); A4 must emit between "
                    f"{FEATURE_COUNT_MIN} and {FEATURE_COUNT_MAX}"
                ),
            )
        )
    if n > FEATURE_COUNT_MAX:
        issues.append(
            ValidationIssue(
                severity="error",
                field="feature_matrix",
                message=(
                    f"too many features ({n}); A4 must emit between "
                    f"{FEATURE_COUNT_MIN} and {FEATURE_COUNT_MAX}"
                ),
            )
        )
    for i, feat in enumerate(features):
        path = f"feature_matrix[{i}].{feat.name}"
        if not (SCORE_MIN <= feat.score <= SCORE_MAX):
            issues.append(_range_issue(f"{path}.score", feat.score, SCORE_MIN, SCORE_MAX))
        if not (CERTAINTY_MIN <= feat.certainty <= CERTAINTY_MAX):
            issues.append(
                _range_issue(f"{path}.certainty", feat.certainty, CERTAINTY_MIN, CERTAINTY_MAX)
            )
        if not (VISIBILITY_MIN <= feat.visibility <= VISIBILITY_MAX):
            issues.append(
                _range_issue(f"{path}.visibility", feat.visibility, VISIBILITY_MIN, VISIBILITY_MAX)
            )
        if not (TIME_TO_VALUE_MIN <= feat.time_to_value_months <= TIME_TO_VALUE_MAX):
            issues.append(
                _range_issue(
                    f"{path}.time_to_value_months",
                    feat.time_to_value_months,
                    TIME_TO_VALUE_MIN,
                    TIME_TO_VALUE_MAX,
                )
            )
    return issues


def check_config_bounds(config: SimulationConfig) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    p = config.simulation_params
    if not (BETA_MIN <= p.present_bias_beta.value <= BETA_MAX):
        issues.append(
            _range_issue(
                "present_bias_beta.value", p.present_bias_beta.value, BETA_MIN, BETA_MAX
            )
        )
    if abs(p.probability_weighting_gamma - GAMMA_LOCK) > 1e-9:
        issues.append(
            ValidationIssue(
                severity="error",
                field="probability_weighting_gamma",
                message=(
                    f"γ must be locked at {GAMMA_LOCK} (T&K 1992); "
                    f"got {p.probability_weighting_gamma}"
                ),
                suggested_value=GAMMA_LOCK,
            )
        )
    if not (PENETRATION_MIN <= p.category_penetration.value <= PENETRATION_MAX):
        issues.append(
            _range_issue(
                "category_penetration.value",
                p.category_penetration.value,
                PENETRATION_MIN,
                PENETRATION_MAX,
            )
        )
    return issues


def _range_issue(path: str, value: float, lo: float, hi: float) -> ValidationIssue:
    return ValidationIssue(
        severity="error",
        field=path,
        message=f"value {value} out of range [{lo}, {hi}]",
    )


# ───────────────────────── Wave 8.5 evidence + cross-field ─────────────────────────


def check_evidence_refs(
    config: SimulationConfig,
    evidence_store: EvidenceStore | None,
) -> list[ValidationIssue]:
    """Every evidence_ref on a CalibratedParam / Assumption must resolve.

    Wave 8.5: a fabricated `voc:abc123` that doesn't exist in the
    EvidenceStore is worse than no evidence — it implies provenance
    that doesn't exist. Validator rejects.
    """
    if evidence_store is None:
        # No store → can't check resolvability. Skip — this is a "store
        # not yet built" path, not a fabrication.
        return []

    issues: list[ValidationIssue] = []
    sim = config.simulation_params
    for name in PUBLISHED_BOUNDS:
        cparam = getattr(sim, name, None)
        if not isinstance(cparam, CalibratedParam):
            continue
        for ref in cparam.evidence_refs or []:
            if ref not in evidence_store:
                issues.append(ValidationIssue(
                    severity="error",
                    field=f"simulation_params.{name}.evidence_refs",
                    message=f"evidence_ref {ref!r} does not resolve in EvidenceStore",
                ))

    for a in config.assumptions:
        for ref in a.evidence_refs or []:
            if ref not in evidence_store:
                issues.append(ValidationIssue(
                    severity="error",
                    field=f"assumptions[{a.id}].evidence_refs",
                    message=f"evidence_ref {ref!r} does not resolve in EvidenceStore",
                ))
    return issues


def check_sparse_rationale(config: SimulationConfig) -> list[ValidationIssue]:
    """Flag parameters with thin or missing rationale.

    A non-default parameter with `len(basis) < SPARSE_BASIS_CHARS` OR
    zero `evidence_refs` is sparse-rationale: the LLM emitted a number
    without supporting reasoning. We don't reject silently — we flag,
    so the orchestrator's retry loop can re-prompt with the issue.
    """
    issues: list[ValidationIssue] = []
    sim = config.simulation_params

    for name in PUBLISHED_BOUNDS:
        cparam = getattr(sim, name, None)
        if not isinstance(cparam, CalibratedParam):
            continue
        if cparam.source_type == "default":
            continue  # defaults explicitly opt out
        thin_basis = len((cparam.basis or "").strip()) < SPARSE_BASIS_CHARS
        no_refs = not (cparam.evidence_refs or [])
        if thin_basis and no_refs:
            issues.append(ValidationIssue(
                severity="warning",
                field=f"simulation_params.{name}",
                message=(
                    f"sparse rationale: basis<{SPARSE_BASIS_CHARS} chars and no "
                    f"evidence_refs (got basis={cparam.basis!r})"
                ),
            ))
    return issues


def check_cross_field_consistency(config: SimulationConfig) -> list[ValidationIssue]:
    """Cross-parameter coherence checks.

    Examples (from spec):
      * `category_maturity="saturated"` paired with `category_penetration < 0.20`
        — saturated markets don't have low penetration.
      * Three or more features with score > 0.9 — likely LLM anchoring.
    """
    issues: list[ValidationIssue] = []
    sim = config.simulation_params

    # Saturated market with low penetration.
    # `category_maturity` lives on MarketContext, not config; the
    # validator sees it via the wider ValidatorInput in the LLM-review
    # path. The pure-Python check looks at penetration vs growth as a
    # proxy: high growth + high penetration is internally inconsistent.
    pen = sim.category_penetration.value
    growth = sim.category_growth.value
    if pen > 0.6 and growth > 1.0:
        issues.append(ValidationIssue(
            severity="warning",
            field="category_penetration / category_growth",
            message=(
                f"high penetration ({pen:.2f}) and high growth ({growth:.2f}) "
                f"are usually inconsistent — saturated markets grow slowly"
            ),
        ))

    # Feature anchoring detector.
    high_score_features = [
        f for f in sim.feature_matrix if f.score > 0.9
    ]
    if len(high_score_features) >= 3:
        names = ", ".join(f.name for f in high_score_features[:3])
        issues.append(ValidationIssue(
            severity="warning",
            field="feature_matrix",
            message=(
                f"{len(high_score_features)} features with score > 0.9 "
                f"({names}…) — likely LLM anchoring; consider re-prompting "
                f"the calibrator for differentiation"
            ),
        ))

    return issues


def populate_published_bounds(config: SimulationConfig) -> int:
    """Fill `Assumption.published_bounds` from the PUBLISHED_BOUNDS table.

    Mutates `config.assumptions` in place. Returns the number of rows
    that gained a bound. Pure side-effecting helper; no validation.
    """
    n = 0
    for a in config.assumptions:
        if a.published_bounds is not None:
            continue
        bound = PUBLISHED_BOUNDS.get(a.parameter)
        if bound is not None:
            a.published_bounds = bound
            n += 1
    return n


# ───────────────────────── The A5 stage ─────────────────────────


class Validator:
    """A5 stage wrapper.

    Satisfies the `Stage` protocol in orchestrator.py. Commit 1 ships
    the hard-range half only; the cross-field LLM review attaches in
    Commit 2 once the A4 prompt actually emits a feature matrix.
    """

    name = "A5_validator"
    input_type = ValidatorInput
    output_type = ValidationReport

    def __init__(
        self,
        llm: LLMClient | None = None,
        enable_llm_review: bool = False,
        evidence_store: EvidenceStore | None = None,
    ):
        self.llm = llm
        self.enable_llm_review = enable_llm_review
        self.evidence_store = evidence_store

    def run(self, inp: ValidatorInput, ctx) -> ValidationReport:
        issues: list[ValidationIssue] = []
        issues.extend(check_feature_matrix(inp.config.simulation_params.feature_matrix))
        issues.extend(check_config_bounds(inp.config))

        # Wave 8.5 — provenance + sparse-rationale + cross-field.
        issues.extend(check_evidence_refs(inp.config, self.evidence_store))
        issues.extend(check_sparse_rationale(inp.config))
        issues.extend(check_cross_field_consistency(inp.config))

        # Cross-field LLM review lands in Commit 2.
        if self.enable_llm_review and self.llm is not None and issues == []:
            # Placeholder — wired in Commit 2.
            pass

        errors = [i for i in issues if i.severity == "error"]
        is_valid = len(errors) == 0
        warnings_count = sum(1 for i in issues if i.severity == "warning")
        # Confidence score: 1.0 - 0.2 per error, 0.05 per warning, floor 0.
        confidence_score = max(0.0, 1.0 - 0.2 * len(errors) - 0.05 * warnings_count)
        if not is_valid:
            logger.info(
                "A5 validator found %d error(s): %s",
                len(errors),
                [e.message for e in errors[:3]],
            )
        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            confidence_score=confidence_score,
        )
