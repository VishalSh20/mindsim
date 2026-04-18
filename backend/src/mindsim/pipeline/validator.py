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
from mindsim.models.config import SimulationConfig
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

    def __init__(self, llm: LLMClient | None = None, enable_llm_review: bool = False):
        self.llm = llm
        self.enable_llm_review = enable_llm_review

    def run(self, inp: ValidatorInput, ctx) -> ValidationReport:
        issues: list[ValidationIssue] = []
        issues.extend(check_feature_matrix(inp.config.simulation_params.feature_matrix))
        issues.extend(check_config_bounds(inp.config))

        # Cross-field LLM review lands in Commit 2.
        if self.enable_llm_review and self.llm is not None and issues == []:
            # Placeholder — wired in Commit 2.
            pass

        errors = [i for i in issues if i.severity == "error"]
        is_valid = len(errors) == 0
        if not is_valid:
            logger.info(
                "A5 validator found %d error(s): %s",
                len(errors),
                [e.message for e in errors[:3]],
            )
        return ValidationReport(is_valid=is_valid, issues=issues)
