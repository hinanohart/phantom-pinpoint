"""Tests for the statistics layer."""

from __future__ import annotations

import numpy as np
import pytest

from phantom_pinpoint.statistics import (
    benjamini_hochberg,
    bootstrap_ci,
    permutation_test,
)


class TestBootstrap:
    def test_zero_centred_data_includes_zero(self) -> None:
        data = np.random.default_rng(0).normal(0.0, 1.0, size=500)
        res = bootstrap_ci(data, seed=7)
        assert res.ci_lo < 0 < res.ci_hi
        assert not res.is_significant
        assert res.n == 500
        assert res.p_two_sided > 0.1

    def test_strong_positive_signal(self) -> None:
        data = np.random.default_rng(0).normal(2.0, 0.5, size=500)
        res = bootstrap_ci(data, seed=7)
        assert res.ci_lo > 0
        assert res.is_significant
        assert res.p_two_sided < 0.05

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([]))

    @pytest.mark.parametrize("ci", [-0.1, 0.0, 1.0, 1.1])
    def test_bad_ci(self, ci: float) -> None:
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([1.0, 2.0]), ci=ci)

    def test_too_few_resamples(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([1.0, 2.0]), n_resamples=10)


class TestPermutationTest:
    def test_difference_of_means(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, size=200)
        b = rng.normal(1.0, 1.0, size=200)
        stat, p = permutation_test(a, b, n_resamples=2000, seed=1)
        assert stat < 0
        assert p < 0.05

    def test_no_difference(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, size=200)
        b = rng.normal(0.0, 1.0, size=200)
        _stat, p = permutation_test(a, b, n_resamples=2000, seed=1)
        assert p > 0.1


class TestBH:
    def test_all_null(self) -> None:
        out = benjamini_hochberg(np.array([0.5, 0.6, 0.9]))
        assert not out.any()

    def test_clear_signals_rejected(self) -> None:
        out = benjamini_hochberg(np.array([0.001, 0.002, 0.6]))
        assert out[0] and out[1] and not out[2]

    def test_empty_input(self) -> None:
        out = benjamini_hochberg(np.array([]))
        assert out.shape == (0,)
