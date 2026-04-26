"""Property-based tests for the ΔPP decomposition (L1 prevention)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from phantom_pinpoint.core import EPS, pp_divergence
from phantom_pinpoint.decomposition import (
    DECOMPOSITION_TOLERANCE,
    DecompositionResult,
    decompose_delta_pp,
    decompose_simulation,
)

_FINITE = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
_POS = st.floats(min_value=0.05, max_value=2.0, allow_nan=False, allow_infinity=False)


def _vec(strat=_FINITE, dim: int = 2) -> st.SearchStrategy[list[float]]:
    return st.lists(strat, min_size=dim, max_size=dim)


@settings(deadline=None, max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    p_star=_vec(), anchor=_vec(), bayes_mean=_vec(),
    sigma_conf=_POS, bayes_var=_POS,
)
def test_sum_to_whole_identity(
    p_star: list[float],
    anchor: list[float],
    bayes_mean: list[float],
    sigma_conf: float,
    bayes_var: float,
) -> None:
    """Width + Location must equal the full log-Bayes-factor."""
    p = np.array(p_star, dtype=np.float64)
    a = np.array(anchor, dtype=np.float64)
    bm = np.array(bayes_mean, dtype=np.float64)
    res = decompose_delta_pp(p, sigma_conf, a, bm, bayes_var)
    legacy = pp_divergence(p, sigma_conf, a, bm, bayes_var)
    assert abs(res.width + res.location - res.delta_pp) < DECOMPOSITION_TOLERANCE
    # And the decomposition recovers the legacy ΔPP up to numerical noise.
    assert abs(res.delta_pp - legacy) < 1e-9 * max(abs(legacy), 1.0)


def test_width_positive_when_strategic_sharper() -> None:
    res = decompose_delta_pp(
        p_star=np.array([0.0, 0.0]),
        sigma_conf=0.1,           # sharp
        anchor=np.array([0.0, 0.0]),
        bayes_mean=np.array([0.0, 0.0]),
        bayes_var=1.0,             # vague
    )
    assert res.width > 0
    assert abs(res.location) < 1e-9
    assert res.delta_pp == pytest.approx(res.width)


def test_location_negative_when_far_from_anchor() -> None:
    res = decompose_delta_pp(
        p_star=np.array([2.0, 0.0]),
        sigma_conf=0.1,
        anchor=np.array([0.0, 0.0]),
        bayes_mean=np.array([2.0, 0.0]),
        bayes_var=0.1,
    )
    # Claim is at bayes_mean, far from anchor → location must be very negative.
    assert res.location < 0
    assert res.delta_pp < 0


def test_decompose_simulation_columns() -> None:
    from phantom_pinpoint.core import PhantomPinpointModel

    model = PhantomPinpointModel(audience_size=2.0)
    res = model.simulate(n_runs=64, seed=7)
    rng = np.random.default_rng(7)
    # Reproduce the prior_means/betas the simulator drew (must mirror the
    # *exact* call sequence in PhantomPinpointModel.simulate).
    n = 64
    _alphas = rng.beta(*model.alpha_dist, size=n)
    betas = rng.beta(*model.beta_dist, size=n)
    _gammas = rng.beta(*model.gamma_dist, size=n) * model.audience_size
    prior_means = np.asarray(model.mu_g, dtype=np.float64) + rng.normal(
        0.0, np.sqrt(model.prior_var), size=(n, model.dim),
    )
    df = decompose_simulation(
        res,
        mu_g=np.asarray(model.mu_g),
        r_g=model.r_g,
        prior_means=prior_means,
        betas=betas,
    )
    assert {"delta_pp", "delta_pp_width", "delta_pp_location"}.issubset(df.columns)
    assert len(df) == n
    # Identity should still hold per row, modulo back-out of bayes_var.
    residual = (df["delta_pp"] - df["delta_pp_width"] - df["delta_pp_location"]).abs()
    assert (residual < DECOMPOSITION_TOLERANCE * np.maximum(df["delta_pp"].abs(), 1.0)).all()


@pytest.mark.parametrize("bad_sigma", [0.0, -0.1])
def test_invalid_sigma_raises(bad_sigma: float) -> None:
    with pytest.raises(ValueError):
        decompose_delta_pp(
            p_star=np.zeros(2),
            sigma_conf=bad_sigma,
            anchor=np.zeros(2),
            bayes_mean=np.zeros(2),
            bayes_var=1.0,
        )


def test_decomposition_result_post_init_does_not_raise_on_noise() -> None:
    """Within tolerance the dataclass must accept the provided values."""
    DecompositionResult(delta_pp=1.0, width=0.5, location=0.5 + EPS)
