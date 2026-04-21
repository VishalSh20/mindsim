"""Wave 5: heuristic maturity classifier tests."""
from __future__ import annotations

import pytest

from mindsim.engine.maturity import (
    classify_from_signals,
    consideration_threshold_for_maturity,
)


class TestClassifyFromSignals:
    def test_nascent(self):
        # Brain-computer interface consumer device — few competitors, little chatter.
        assert classify_from_signals(n_competitors=1, review_volume=30) == "nascent"
        assert classify_from_signals(n_competitors=2, review_volume=80) == "nascent"

    def test_growing(self):
        # AI coding assistant — handful of competitors, moderate chatter.
        assert classify_from_signals(n_competitors=4, review_volume=2000) == "growing"
        assert classify_from_signals(n_competitors=5, review_volume=500) == "growing"

    def test_mainstream(self):
        # Habit tracker — many competitors, plenty of reviews.
        assert classify_from_signals(n_competitors=8, review_volume=15_000) == "mainstream"
        assert classify_from_signals(n_competitors=12, review_volume=40_000) == "mainstream"

    def test_saturated(self):
        # CRM — huge chatter crosses saturation cap regardless of competitor count.
        assert classify_from_signals(n_competitors=20, review_volume=200_000) == "saturated"
        assert classify_from_signals(n_competitors=3, review_volume=60_000) == "saturated"

    def test_age_hint_promotes_legacy_category(self):
        """`established` hint nudges growing → mainstream."""
        base = classify_from_signals(n_competitors=4, review_volume=2000)
        assert base == "growing"
        with_hint = classify_from_signals(
            n_competitors=4, review_volume=2000,
            category_age_hint="established spreadsheet tooling since the 1990s",
        )
        assert with_hint == "mainstream"

    def test_age_hint_demotes_new_category(self):
        """`new / 2025` hint nudges mainstream → growing."""
        base = classify_from_signals(n_competitors=8, review_volume=15_000)
        assert base == "mainstream"
        with_hint = classify_from_signals(
            n_competitors=8, review_volume=15_000,
            category_age_hint="new AI agents 2025",
        )
        assert with_hint == "growing"

    def test_non_negative_inputs(self):
        # Negative inputs coerced to 0, not raised.
        assert classify_from_signals(n_competitors=-3, review_volume=-5) == "nascent"


class TestConsiderationThresholdMapping:
    def test_each_bucket_resolves(self):
        # Regression guard — buckets from classify must all map.
        for m in ("nascent", "growing", "mainstream", "saturated"):
            t = consideration_threshold_for_maturity(m)
            assert 0.0 < t < 1.0

    def test_unknown_uses_default(self):
        t_none = consideration_threshold_for_maturity(None)
        t_bad = consideration_threshold_for_maturity("bogus")
        assert t_none == t_bad
