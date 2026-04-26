"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed RNG (seed=42) used in deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def small_baseline_kwargs() -> dict[str, object]:
    """Minimal but realistic kwargs that yield ΔPP > 0 in baseline."""
    return {
        "dim": 2,
        "mu_g": (0.0, 0.0),
        "r_g": 1.0,
        "g_sigma": 1.5,
        "tau": 0.3,
        "prior_var": 4.0,
        "audience_size": 1.0,
    }
