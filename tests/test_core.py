"""Unit tests for the mathematical core."""

from __future__ import annotations

import numpy as np
import pytest

from phantom_pinpoint.core import (
    EPS,
    PhantomPinpointModel,
    bayesian_posterior_mean,
    confabulation_width,
    pp_divergence,
    project_onto_region,
    strategic_claim,
)


class TestProjection:
    def test_inside_ball_unchanged(self) -> None:
        x = np.array([0.3, 0.4])
        out = project_onto_region(x, np.zeros(2), radius=1.0)
        np.testing.assert_allclose(out, x)

    def test_outside_pulled_to_surface(self) -> None:
        x = np.array([3.0, 4.0])  # |x| = 5
        out = project_onto_region(x, np.zeros(2), radius=1.0)
        np.testing.assert_allclose(out, np.array([0.6, 0.8]), atol=1e-12)
        assert np.isclose(np.linalg.norm(out), 1.0)

    def test_batched(self) -> None:
        pts = np.array([[5.0, 0.0], [0.5, 0.0], [-2.0, 2.0]])
        out = project_onto_region(pts, np.zeros(2), radius=1.0)
        norms = np.linalg.norm(out, axis=1)
        # First and third are pulled to the surface; second was already inside.
        np.testing.assert_allclose(norms, [1.0, 0.5, 1.0], atol=1e-12)

    @pytest.mark.parametrize("bad_radius", [0.0, -1.0])
    def test_bad_radius_raises(self, bad_radius: float) -> None:
        with pytest.raises(ValueError):
            project_onto_region(np.zeros(2), np.zeros(2), radius=bad_radius)


class TestBayesianPosterior:
    def test_conjugate_formula(self) -> None:
        mu, post_var = bayesian_posterior_mean(
            np.array([0.0, 0.0]), prior_var=1.0,
            trigger=np.array([2.0, 0.0]), likelihood_var=1.0,
        )
        # Equal precisions → posterior is the midpoint.
        np.testing.assert_allclose(mu, [1.0, 0.0])
        assert np.isclose(post_var, 0.5)

    def test_strong_likelihood_dominates(self) -> None:
        mu, post_var = bayesian_posterior_mean(
            np.zeros(2), 100.0, np.array([1.0, 0.0]), 0.01,
        )
        np.testing.assert_allclose(mu, [1.0, 0.0], atol=1e-3)
        assert post_var < 0.011

    @pytest.mark.parametrize("bad", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
    def test_bad_variances(self, bad: tuple[float, float]) -> None:
        with pytest.raises(ValueError):
            bayesian_posterior_mean(np.zeros(2), bad[0], np.zeros(2), bad[1])


class TestConfabulationWidth:
    def test_monotone_in_alpha(self) -> None:
        widths = [confabulation_width(a, 0.0) for a in [0.0, 0.5, 1.0, 5.0]]
        assert widths == sorted(widths, reverse=True)

    def test_monotone_in_gamma(self) -> None:
        widths = [confabulation_width(0.5, g) for g in [0.0, 1.0, 5.0]]
        assert widths == sorted(widths, reverse=True)

    def test_floor_respected(self) -> None:
        w = confabulation_width(1e6, 1e6, sigma_floor=0.1, sigma_naive=0.5)
        assert w == 0.1

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            confabulation_width(-1.0, 0.0)


class TestStrategicClaim:
    def test_anchor_is_convex_combination(self, rng: np.random.Generator) -> None:
        E = np.array([2.0, 0.0])
        prior = np.array([0.0, 0.0])
        p_star, sigma, anchor = strategic_claim(
            E, prior, mu_g=np.zeros(2), r_g=1.0,
            alpha=10.0, beta=0.5, gamma=10.0, rng=rng,
        )
        # alpha+gamma large → sigma at floor → claim very near anchor.
        proj = np.array([1.0, 0.0])  # ||E||=2, projects to ball of radius 1.
        expected_anchor = 0.5 * proj + 0.5 * prior  # beta=0.5 mid
        np.testing.assert_allclose(anchor, expected_anchor)
        np.testing.assert_allclose(p_star, expected_anchor, atol=0.2)
        assert sigma > 0

    def test_seed_determinism(self) -> None:
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        E = np.array([1.0, 1.0])
        a, _, _ = strategic_claim(E, np.zeros(2), np.zeros(2), 1.0, 1.0, 0.3, 1.0, rng_a)
        b, _, _ = strategic_claim(E, np.zeros(2), np.zeros(2), 1.0, 1.0, 0.3, 1.0, rng_b)
        np.testing.assert_array_equal(a, b)


class TestPPDivergence:
    def test_positive_when_claim_at_anchor(self) -> None:
        # Claim sits exactly at strategic anchor; bayes_mean is elsewhere —
        # log-Bayes-factor must be positive.
        delta = pp_divergence(
            p_star=np.array([0.5, 0.0]),
            sigma_conf=0.1,
            anchor=np.array([0.5, 0.0]),
            bayes_mean=np.array([2.0, 0.0]),
            bayes_var=0.1,
        )
        assert delta > 0

    def test_negative_when_claim_at_bayes_mean(self) -> None:
        # Claim coincides with bayes_mean but anchor is elsewhere — Bayesian
        # model wins and ΔPP < 0.
        delta = pp_divergence(
            p_star=np.array([2.0, 0.0]),
            sigma_conf=0.1,
            anchor=np.array([0.5, 0.0]),
            bayes_mean=np.array([2.0, 0.0]),
            bayes_var=0.1,
        )
        assert delta < 0

    def test_symmetric_when_anchor_equals_post(self) -> None:
        # If anchor and bayes_mean are identical and widths agree, ΔPP = 0
        # for any claim (model un-identifiable in this degenerate case).
        delta = pp_divergence(
            p_star=np.array([0.3, 0.4]),
            sigma_conf=0.5,
            anchor=np.array([0.0, 0.0]),
            bayes_mean=np.array([0.0, 0.0]),
            bayes_var=0.25,
        )
        assert abs(delta) < EPS * 1000


class TestPhantomPinpointModel:
    def test_validation_dim(self) -> None:
        with pytest.raises(ValueError):
            PhantomPinpointModel(dim=0)

    def test_validation_mu_g_length(self) -> None:
        with pytest.raises(ValueError):
            PhantomPinpointModel(dim=2, mu_g=(0.0, 0.0, 0.0))

    def test_simulate_determinism(self) -> None:
        m = PhantomPinpointModel()
        a = m.simulate(n_runs=64, seed=123)
        b = m.simulate(n_runs=64, seed=123)
        np.testing.assert_array_equal(a.delta_pp, b.delta_pp)
        np.testing.assert_array_equal(a.p_star, b.p_star)

    def test_baseline_positive_delta_pp(self) -> None:
        m = PhantomPinpointModel(audience_size=2.0)
        res = m.simulate(n_runs=400, seed=42)
        # Sanity: baseline produces positive average ΔPP — not a strong test
        # but enough to catch sign-flip regressions in pp_divergence.
        assert float(res.delta_pp.mean()) > 0.0

    def test_bayesian_baseline_drops_signal(self) -> None:
        strategic = PhantomPinpointModel(audience_size=2.0).simulate(400, seed=42)
        bayes = PhantomPinpointModel(
            audience_size=2.0, use_bayesian_baseline=True
        ).simulate(400, seed=42)
        # Negative control — bayesian should be strictly weaker.
        assert float(bayes.delta_pp.mean()) < float(strategic.delta_pp.mean())

    def test_shuffle_g_reduces_post_hoc_fit(self) -> None:
        baseline = PhantomPinpointModel().simulate(400, seed=42)
        shuf = PhantomPinpointModel(shuffle_g=True).simulate(400, seed=42)
        assert float(shuf.post_hoc_fit.mean()) <= float(baseline.post_hoc_fit.mean())
