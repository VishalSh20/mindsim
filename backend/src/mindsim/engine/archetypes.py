"""Archetype definitions — Rogers (1962) adoption curve segments.

Loads archetype configurations from YAML and provides typed access
to population shares and parameter distributions.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ParamDistribution:
    """Normal distribution parameters for an agent attribute."""

    mean: float
    std: float
    min: float = 0.0
    max: float = float("inf")


@dataclass
class ArchetypeAwareness:
    """Awareness levels for an archetype."""

    product: float = 0.5
    competitor_fraction: float = 0.5


@dataclass
class ArchetypeConfig:
    """Complete configuration for one archetype."""

    name: str
    share: float
    label: str
    description: str
    distributions: dict[str, ParamDistribution] = field(default_factory=dict)
    awareness: ArchetypeAwareness = field(default_factory=ArchetypeAwareness)


@dataclass
class LossAversionMixture:
    """Gächter et al. 2022 mixture model parameters."""

    archetype_weight: float = 0.80
    near_zero_weight: float = 0.20
    near_zero_mean: float = 1.1
    near_zero_std: float = 0.2


@dataclass
class Correlations:
    """Personality-economics correlation coefficients."""

    neuroticism_loss_aversion: float = 0.3
    openness_novelty: float = 0.3
    agreeableness_social_proof: float = 0.4


@dataclass
class ArchetypeSet:
    """All archetypes + global parameters."""

    archetypes: dict[str, ArchetypeConfig] = field(default_factory=dict)
    loss_aversion_mixture: LossAversionMixture = field(
        default_factory=LossAversionMixture
    )
    correlations: Correlations = field(default_factory=Correlations)

    @property
    def names(self) -> list[str]:
        return list(self.archetypes.keys())

    @property
    def shares(self) -> list[float]:
        return [a.share for a in self.archetypes.values()]


def _get_config_dir() -> Path:
    """Get the config directory path."""
    return Path(__file__).parent.parent / "config"


def load_archetypes(path: Path | None = None) -> ArchetypeSet:
    """Load archetype definitions from YAML.

    Args:
        path: Path to archetypes.yaml. If None, uses the bundled config.

    Returns:
        ArchetypeSet with all archetype configurations.
    """
    if path is None:
        path = _get_config_dir() / "archetypes.yaml"

    with open(path) as f:
        data = yaml.safe_load(f)

    archetypes = {}
    for name, aconf in data["archetypes"].items():
        distributions = {}
        for param_name, pdist in aconf["distributions"].items():
            distributions[param_name] = ParamDistribution(
                mean=pdist["mean"],
                std=pdist["std"],
                min=pdist.get("min", 0.0),
                max=pdist.get("max", float("inf")),
            )

        awareness = ArchetypeAwareness(
            product=aconf["awareness"]["product"],
            competitor_fraction=aconf["awareness"]["competitor_fraction"],
        )

        archetypes[name] = ArchetypeConfig(
            name=name,
            share=aconf["share"],
            label=aconf["label"],
            description=aconf["description"],
            distributions=distributions,
            awareness=awareness,
        )

    mixture_data = data.get("loss_aversion_mixture", {})
    mixture = LossAversionMixture(
        archetype_weight=mixture_data.get("archetype_weight", 0.80),
        near_zero_weight=mixture_data.get("near_zero_weight", 0.20),
        near_zero_mean=mixture_data.get("near_zero_mean", 1.1),
        near_zero_std=mixture_data.get("near_zero_std", 0.2),
    )

    corr_data = data.get("correlations", {})
    correlations = Correlations(
        neuroticism_loss_aversion=corr_data.get("neuroticism_loss_aversion", 0.3),
        openness_novelty=corr_data.get("openness_novelty", 0.3),
        agreeableness_social_proof=corr_data.get("agreeableness_social_proof", 0.4),
    )

    return ArchetypeSet(
        archetypes=archetypes,
        loss_aversion_mixture=mixture,
        correlations=correlations,
    )
