"""Tests for the identifiability degeneracy diagnostic (AC9)."""

from __future__ import annotations

import numpy as np
import pytest

from phantom_pinpoint.identifiability import (
    DEFAULT_PRIOR_PROXIMITY,
    DegeneracyReport,
    assess,
    detect_degeneracy,
)


def test_inside_g_and_prior_close_flagged() -> None:
    triggers = np.array([[0.1, 0.0], [3.0, 0.0]])
    priors = np.array([[0.05, 0.0], [3.0, 0.0]])
    mask = detect_degeneracy(triggers, priors, mu_g=np.zeros(2), r_g=1.0)
    assert mask[0]
    assert not mask[1]  # trigger outside G


def test_far_prior_not_flagged() -> None:
    triggers = np.array([[0.0, 0.0]])
    priors = np.array([[0.5, 0.0]])  # distance 0.5 > 0.1 * 1.0
    mask = detect_degeneracy(triggers, priors, mu_g=np.zeros(2), r_g=1.0)
    assert not mask[0]


def test_assess_returns_report() -> None:
    rng = np.random.default_rng(0)
    delta_pp = rng.normal(loc=0.0, scale=1.0, size=200)
    is_deg = np.zeros(200, dtype=np.bool_)
    is_deg[:50] = True
    rep = assess(delta_pp, is_deg, n_resamples=500, seed=1)
    assert isinstance(rep, DegeneracyReport)
    assert 0.0 <= rep.fraction_degenerate <= 1.0
    assert rep.n_inside == 50
    assert rep.n_outside == 150


def test_assess_underpowered_does_not_raise() -> None:
    delta_pp = np.array([1.0, 2.0, 3.0])
    is_deg = np.array([True, False, False])
    rep = assess(delta_pp, is_deg, seed=1, n_resamples=100)
    assert rep.n_inside == 1
    assert rep.n_outside == 2
    # Both regions undersized (<5) → AC9 cannot pass.
    assert not rep.ac9_passed


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        detect_degeneracy(
            triggers=np.zeros((2, 2)),
            prior_means=np.zeros((3, 2)),
            mu_g=np.zeros(2),
            r_g=1.0,
        )
    with pytest.raises(ValueError):
        detect_degeneracy(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(2), r_g=0.0)
    with pytest.raises(ValueError):
        detect_degeneracy(
            np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(2), r_g=1.0, prior_proximity=0.0,
        )


def test_default_prior_proximity_is_documented() -> None:
    assert DEFAULT_PRIOR_PROXIMITY == 0.1
