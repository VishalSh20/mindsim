"""Numerical tests for the Tversky-Kahneman probability weighting function.

Anchors come from evaluating w(p) = p^γ / (p^γ + (1-p)^γ)^(1/γ) with γ=0.61
by hand; tolerances are loose (± 0.005) so minor numerical drift doesn't
break the test.
"""
import numpy as np
import pytest

from mindsim.engine.probability_weighting import (
    DEFAULT_GAMMA,
    probability_weight,
)


def test_endpoints_are_exact():
    assert probability_weight(0.0) == 0.0
    assert probability_weight(1.0) == 1.0


def test_midpoint():
    # w(0.5) at γ=0.61 ≈ 0.421
    assert probability_weight(0.5) == pytest.approx(0.421, abs=0.005)


def test_small_probabilities_overweighted():
    # w(0.1) ≈ 0.186 > 0.1 → small prob overweighted
    w = probability_weight(0.1)
    assert w == pytest.approx(0.186, abs=0.005)
    assert w > 0.1


def test_large_probabilities_underweighted():
    # w(0.9) ≈ 0.714 < 0.9 → large prob underweighted
    w = probability_weight(0.9)
    assert w == pytest.approx(0.714, abs=0.005)
    assert w < 0.9


def test_monotonically_increasing():
    probs = np.linspace(0, 1, 101)
    weighted = probability_weight(probs)
    # Strict monotonic on interior points.
    assert np.all(np.diff(weighted) >= -1e-12)


def test_vectorised_matches_scalar():
    probs = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    vec = probability_weight(probs)
    for i, p in enumerate(probs):
        assert vec[i] == pytest.approx(probability_weight(float(p)), abs=1e-12)


def test_clips_inputs_outside_unit_interval():
    assert probability_weight(-0.5) == 0.0
    assert probability_weight(1.5) == 1.0


def test_rejects_bad_gamma():
    with pytest.raises(ValueError):
        probability_weight(0.5, gamma=0.0)
    with pytest.raises(ValueError):
        probability_weight(0.5, gamma=-0.1)
    with pytest.raises(ValueError):
        probability_weight(0.5, gamma=1.5)


def test_default_gamma_is_061():
    assert DEFAULT_GAMMA == 0.61


def test_preserves_scalar_vs_array_shape():
    # Scalar in → scalar (float) out
    assert isinstance(probability_weight(0.5), float)
    # Array in → array out, same shape
    arr = np.array([[0.1, 0.5], [0.7, 0.9]])
    result = probability_weight(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == arr.shape
