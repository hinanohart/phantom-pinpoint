"""Tests for effect_size estimators."""

from __future__ import annotations

import numpy as np
import pytest

from phantom_pinpoint.effect_size import (
    EffectSizeCI,
    bootstrap_effect_size,
    cliffs_delta,
    cohens_d,
    hedges_g,
)


class TestCohensD:
    def test_zero_when_identical(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=100)
        assert abs(cohens_d(x, x.copy())) < 1e-12

    def test_one_for_unit_separation(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(loc=0.0, scale=1.0, size=2000)
        b = rng.normal(loc=1.0, scale=1.0, size=2000)
        d = cohens_d(a, b)
        assert -1.05 < d < -0.95

    def test_paired_branch(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        y = x + 0.5 + rng.normal(scale=0.1, size=200)
        d = cohens_d(x, y, paired=True)
        # |y - x| ≈ 0.5, std ≈ 0.1 → d ≈ -5
        assert d < -4

    def test_paired_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            cohens_d(np.zeros(5), np.zeros(7), paired=True)

    def test_zero_variance(self) -> None:
        # Two constant series of equal value → d = 0
        a = np.ones(50)
        b = np.ones(50)
        assert cohens_d(a, b) == 0.0

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            cohens_d(np.array([]), np.zeros(5))


class TestHedgesG:
    def test_correction_smaller_than_d(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, size=10)
        b = rng.normal(1.0, 1.0, size=10)
        d = cohens_d(a, b)
        g = hedges_g(a, b)
        assert abs(g) < abs(d)
        # Bias factor approaches 1 for large n.
        a_large = rng.normal(0.0, 1.0, size=10_000)
        b_large = rng.normal(1.0, 1.0, size=10_000)
        d_large = cohens_d(a_large, b_large)
        g_large = hedges_g(a_large, b_large)
        assert abs(g_large - d_large) < 1e-3


class TestCliffsDelta:
    def test_bounds(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, size=200)
        b = rng.normal(2.0, 1.0, size=200)
        delta = cliffs_delta(a, b)
        assert -1.0 <= delta <= 1.0
        assert delta < -0.5

    def test_no_difference_close_to_zero(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, size=500)
        b = rng.normal(0.0, 1.0, size=500)
        delta = cliffs_delta(a, b)
        assert abs(delta) < 0.15


class TestBootstrapEffectSize:
    def test_basic_signal(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, size=200)
        b = rng.normal(1.0, 1.0, size=200)
        ci = bootstrap_effect_size(a, b, statistic=cohens_d, n_resamples=2000, seed=7)
        assert isinstance(ci, EffectSizeCI)
        assert ci.is_significant
        assert ci.ci_hi < 0  # a < b

    def test_no_signal_ci_contains_zero(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, size=300)
        b = rng.normal(0.0, 1.0, size=300)
        ci = bootstrap_effect_size(a, b, statistic=cohens_d, n_resamples=2000, seed=7)
        assert not ci.is_significant
        assert ci.ci_lo < 0 < ci.ci_hi

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_effect_size(np.array([1.0]), np.array([1.0, 2.0]))
        with pytest.raises(ValueError):
            bootstrap_effect_size(np.zeros(5), np.zeros(5), n_resamples=10)
        with pytest.raises(ValueError):
            bootstrap_effect_size(np.zeros(5), np.zeros(5), ci=1.5)

    def test_as_dict_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(size=50)
        b = rng.normal(loc=1.0, size=50)
        ci = bootstrap_effect_size(a, b, n_resamples=500, seed=1)
        d = ci.as_dict()
        assert d["name"] == "cohens_d"
        assert d["n_a"] == 50
        assert "is_significant" in d
