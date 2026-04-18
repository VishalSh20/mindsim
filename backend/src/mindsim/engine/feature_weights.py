"""Loader and validator for config/feature_weights.yaml.

Archetype × category weight matrix. Rows must sum to 1.0 (strict).
Categories are the four v2-middle feature categories defined by the
`Feature.category` Literal in models/product.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from mindsim.models.config import ARCHETYPE_NAMES

FEATURE_CATEGORIES: tuple[str, ...] = (
    "core_value",
    "social_signal",
    "ongoing_cost",
    "switching_friction_reducer",
)

DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "config" / "feature_weights.yaml"
)


@dataclass(frozen=True)
class FeatureWeights:
    """Validated archetype × category weight matrix.

    `by_archetype[archetype][category]` gives the base weight. Rows sum
    to exactly 1.0. `noise_sigma` is the multiplicative jitter applied
    to each weight at population generation, before renormalisation.
    """

    by_archetype: dict[str, dict[str, float]]
    noise_sigma: float

    def weights_for(self, archetype: str) -> dict[str, float]:
        if archetype not in self.by_archetype:
            raise KeyError(f"unknown archetype {archetype!r}")
        return dict(self.by_archetype[archetype])

    def categories(self) -> tuple[str, ...]:
        return FEATURE_CATEGORIES


def load_feature_weights(path: Path | None = None) -> FeatureWeights:
    """Load and strictly validate the feature_weights YAML.

    Raises ValueError on any structural defect: missing archetype, wrong
    category set, rows that don't sum to 1.0 within a tight tolerance.
    """
    src = path or DEFAULT_PATH
    with src.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "archetypes" not in raw:
        raise ValueError(f"{src}: missing top-level `archetypes` mapping")

    arche = raw["archetypes"]
    missing = set(ARCHETYPE_NAMES) - set(arche.keys())
    if missing:
        raise ValueError(f"{src}: missing archetypes: {sorted(missing)}")

    by_archetype: dict[str, dict[str, float]] = {}
    for name in ARCHETYPE_NAMES:
        row = arche[name]
        if not isinstance(row, dict):
            raise ValueError(f"{src}: archetype {name!r} must be a mapping")
        missing_cats = set(FEATURE_CATEGORIES) - set(row.keys())
        if missing_cats:
            raise ValueError(
                f"{src}: archetype {name!r} missing categories {sorted(missing_cats)}"
            )
        extra_cats = set(row.keys()) - set(FEATURE_CATEGORIES)
        if extra_cats:
            raise ValueError(
                f"{src}: archetype {name!r} has unknown categories {sorted(extra_cats)}"
            )
        total = sum(float(row[c]) for c in FEATURE_CATEGORIES)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"{src}: archetype {name!r} weights sum to {total:.6f}, expected 1.0"
            )
        for c in FEATURE_CATEGORIES:
            v = float(row[c])
            if v < 0.0 or v > 1.0:
                raise ValueError(
                    f"{src}: archetype {name!r} category {c!r} has invalid weight {v}"
                )
        by_archetype[name] = {c: float(row[c]) for c in FEATURE_CATEGORIES}

    noise_cfg = raw.get("noise", {}) or {}
    sigma = float(noise_cfg.get("sigma", 0.05))
    if sigma < 0.0 or sigma > 0.5:
        raise ValueError(f"{src}: noise.sigma must be in [0, 0.5], got {sigma}")

    return FeatureWeights(by_archetype=by_archetype, noise_sigma=sigma)


def jitter_and_renormalise(
    base: dict[str, float],
    rng,  # np.random.Generator — typed loosely to keep this module numpy-agnostic
    sigma: float,
) -> dict[str, float]:
    """Apply multiplicative noise and renormalise to sum=1.0.

    Returns a new dict; does not mutate `base`.
    """
    import numpy as np

    keys = list(base.keys())
    values = np.array([base[k] for k in keys], dtype=np.float64)
    noise = rng.normal(loc=1.0, scale=sigma, size=values.shape)
    jittered = np.clip(values * noise, 1e-9, None)
    jittered /= jittered.sum()
    return {k: float(v) for k, v in zip(keys, jittered)}
