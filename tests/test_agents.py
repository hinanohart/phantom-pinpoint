"""Tests for the per-agent wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from phantom_pinpoint.agents import Audience, Subject, Trigger


def test_trigger_likelihood_var() -> None:
    t = Trigger(location=np.array([0.0, 0.0]), salience=2.0)
    assert pytest.approx(t.likelihood_var(0.4)) == (0.2) ** 2


def test_trigger_bad_salience() -> None:
    t = Trigger(location=np.zeros(2), salience=0.0)
    with pytest.raises(ValueError):
        t.likelihood_var(0.4)


def test_audience_validation() -> None:
    Audience(size=2, vigilance=0.5)
    with pytest.raises(ValueError):
        Audience(size=-1)
    with pytest.raises(ValueError):
        Audience(size=1, vigilance=2.0)


def test_subject_history_bounded(rng: np.random.Generator) -> None:
    s = Subject(
        dim=2,
        alpha=1.0,
        beta=0.3,
        gamma=1.0,
        prior_mean=np.zeros(2),
        prior_var=4.0,
        mu_g=np.zeros(2),
        r_g=1.0,
        memory_size=4,
    )
    for k in range(10):
        s.claim(Trigger(np.array([float(k), 0.0])), rng=rng)
    assert len(s.history) == 4
    assert all(isinstance(d, float) for _, _, d in s.replay())
