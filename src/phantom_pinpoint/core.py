"""Mathematical core of the Phantom Pinpoint (PP) Effect.

This module implements the five governing equations of the PP generator, as
specified in :doc:`docs/theory.md`:

1. **Prior dynamics** — Ornstein–Uhlenbeck drift of the subject's belief
   :math:`\\psi_t` toward the abstract goal centre :math:`\\mu_G`.
2. **Bayesian baseline posterior** — Gaussian-Gaussian conjugacy used as the
   *rational* counterfactual against which PP confabulation is measured.
3. **Strategic claim** — argmax over a specificity / defensiveness / audience
   reward, producing the claimed point :math:`p^*`.
4. **Confabulation kernel** — Gaussian noise around the closest-point
   projection :math:`\\Pi_G(E)`, with width shrinking under audience pressure.
5. **PP divergence** — Kullback-Leibler divergence-difference identifying when
   the claim is closer to the trigger than to the genuine prior, i.e.
   ":math:`\\Delta_{PP} > 0`" iff the agent is exhibiting Phantom Pinpoint.

The implementation is intentionally pure-numpy (no PyTorch / JAX / Mesa) so
that the simulation is small, auditable and bit-exactly reproducible across
machines given a seeded :class:`numpy.random.Generator`.

Notation reused throughout the code base:

* ``mu_g``: centre of the abstract goal region :math:`G \\subset \\mathbb{R}^d`.
* ``r_g``: Gaussian "radius" (one standard deviation) of :math:`G` — the
  vagueness scale.  ``r_g`` large → vague target → strong PP.
* ``E``: trigger location in :math:`\\mathbb{R}^d`.
* ``alpha`` ∈ [0, 1]: claim-specificity gain (how sharp the post-hoc claim is).
* ``beta``  ∈ [0, 1]: defensive bias (how much p* is anchored to the prior).
* ``gamma``: audience pressure (multiplicative on alpha; γ=0 → private).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

from phantom_pinpoint._logging import get_logger

_LOG = get_logger("core")

#: Numerical floor for variances and claim widths.  Avoids division by zero in
#: KL divergences and entropy calculations without distorting the qualitative
#: behaviour.  Calibrated against the Gaussian density at :math:`\\sqrt{2\\pi}`.
EPS: Final[float] = 1e-9


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def project_onto_region(
    point: NDArray[np.float64],
    centre: NDArray[np.float64],
    radius: float,
) -> NDArray[np.float64]:
    """Closest-point projection of ``point`` onto a Euclidean ball ``B(centre, radius)``.

    Parameters
    ----------
    point:
        Shape ``(d,)`` or ``(n, d)``.  Single vector or batched.
    centre:
        Shape ``(d,)``.  Centre :math:`\\mu_G` of the goal region.
    radius:
        Scalar :math:`r_g > 0`.  Boundary of the Euclidean ball used as a hard
        approximation of the (otherwise soft Gaussian) goal region.

    Returns
    -------
    numpy.ndarray
        Same shape as ``point``.  Points already inside the ball are returned
        unchanged; outside points are pushed onto the ball surface.

    Raises
    ------
    ValueError
        If ``radius <= 0`` or shapes mismatch.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius!r}")
    point = np.asarray(point, dtype=np.float64)
    centre = np.asarray(centre, dtype=np.float64)
    if point.shape[-1] != centre.shape[-1]:
        raise ValueError(
            f"point and centre last-dim mismatch: {point.shape} vs {centre.shape}"
        )
    delta = point - centre
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = np.where(norm > radius, radius / np.maximum(norm, EPS), 1.0)
    return centre + delta * scale


def bayesian_posterior_mean(
    prior_mean: NDArray[np.float64],
    prior_var: float,
    trigger: NDArray[np.float64],
    likelihood_var: float,
) -> tuple[NDArray[np.float64], float]:
    """Closed-form Gaussian-Gaussian posterior under conjugate update.

    Used as the *rational* baseline that the strategic PP claim is compared
    against in :func:`pp_divergence`.

    Parameters
    ----------
    prior_mean:
        ``(d,)`` mean of :math:`\\psi_t = \\mathcal{N}(\\mu, \\sigma_\\psi^2 I)`.
    prior_var:
        Scalar isotropic variance :math:`\\sigma_\\psi^2`.
    trigger:
        ``(d,)`` observation :math:`E` with isotropic likelihood
        :math:`\\mathcal{N}(E; \\theta, \\tau^2 I)`.
    likelihood_var:
        Scalar :math:`\\tau^2`.

    Returns
    -------
    tuple
        ``(posterior_mean, posterior_var)``.

    Notes
    -----
    For isotropic Gaussians,

    .. math::

        \\sigma_{post}^2 = \\bigl(\\sigma_\\psi^{-2} + \\tau^{-2}\\bigr)^{-1},
        \\quad
        \\mu_{post} = \\sigma_{post}^2 (\\mu \\sigma_\\psi^{-2} + E \\tau^{-2}).
    """
    if prior_var <= 0 or likelihood_var <= 0:
        raise ValueError("variances must be positive")
    inv_post = 1.0 / prior_var + 1.0 / likelihood_var
    post_var = 1.0 / inv_post
    post_mean = post_var * (
        np.asarray(prior_mean, dtype=np.float64) / prior_var
        + np.asarray(trigger, dtype=np.float64) / likelihood_var
    )
    return post_mean, float(post_var)


# --------------------------------------------------------------------------- #
# Strategic / confabulation core
# --------------------------------------------------------------------------- #
def confabulation_width(
    alpha: float,
    gamma: float,
    sigma_floor: float = 0.02,
    sigma_naive: float = 0.5,
) -> float:
    """Width :math:`\\sigma_{conf}(\\alpha, \\gamma)` of the post-hoc claim kernel.

    Larger ``alpha`` (specificity gain) and larger ``gamma`` (audience
    pressure) shrink the claim toward a point estimate, reproducing the
    qualitative finding that public claims are *narrower* than private
    rationalisations.

    Parameters
    ----------
    alpha, gamma:
        Non-negative gains.
    sigma_floor:
        Lower bound preventing degenerate :math:`\\delta`-spikes.
    sigma_naive:
        Width when ``alpha = gamma = 0`` (purely Bayesian-like spread).

    Returns
    -------
    float
        :math:`\\sigma_{conf} = \\max(\\sigma_{floor}, \\sigma_{naive} / (1 + \\alpha + \\gamma))`.

    Raises
    ------
    ValueError
        If gains or bounds are negative, or ``sigma_floor > sigma_naive``.
    """
    if alpha < 0 or gamma < 0:
        raise ValueError(f"alpha, gamma must be >= 0, got {alpha=}, {gamma=}")
    if sigma_floor <= 0 or sigma_naive <= 0:
        raise ValueError("sigma_floor and sigma_naive must be > 0")
    if sigma_floor > sigma_naive:
        raise ValueError("sigma_floor must be <= sigma_naive")
    return float(max(sigma_floor, sigma_naive / (1.0 + alpha + gamma)))


def strategic_anchor(
    trigger: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    mu_g: NDArray[np.float64],
    r_g: float,
    beta: float,
) -> NDArray[np.float64]:
    """Convex blend of :math:`\\Pi_G(E)` (closest-point) and the prior mean.

    The closed-form solution to the strategic argmax in Eq. 3 — see
    :func:`strategic_claim` for full discussion.  Exposed separately so that
    :func:`pp_divergence` can reuse the *same* anchor when scoring claims that
    were actually generated by the Bayesian baseline (negative control).
    """
    trigger = np.asarray(trigger, dtype=np.float64)
    prior_mean = np.asarray(prior_mean, dtype=np.float64)
    mu_g = np.asarray(mu_g, dtype=np.float64)
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta!r}")
    proj = project_onto_region(trigger, mu_g, r_g)
    return (1.0 - beta) * proj + beta * prior_mean


def strategic_claim(
    trigger: NDArray[np.float64],
    prior_mean: NDArray[np.float64],
    mu_g: NDArray[np.float64],
    r_g: float,
    alpha: float,
    beta: float,
    gamma: float,
    rng: np.random.Generator,
    sigma_floor: float = 0.02,
    sigma_naive: float = 0.5,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Generate the claimed point :math:`p^*` from a single PP agent.

    Implements the strategic argmax (Eq. 3) in *closed form* by exploiting
    that all reward terms are quadratic / Gaussian-log: the optimum of

    .. math::
        p^* = \\arg\\max_p\\;
            \\alpha\\,\\log \\mathcal{N}(p; \\Pi_G(E), \\sigma_0^2)
          - \\beta\\,\\|p - \\mu\\|^2

    is a convex combination of :math:`\\Pi_G(E)` and the prior mean
    :math:`\\mu`, with audience pressure :math:`\\gamma` *narrowing* the
    sampling kernel (Eq. 4).

    Parameters
    ----------
    trigger:
        ``(d,)`` location :math:`E` of the external trigger.
    prior_mean:
        ``(d,)`` mean :math:`\\mu` of the agent's prior :math:`\\psi_t`.
    mu_g, r_g:
        Centre and (hard) radius of the goal region :math:`G`.
    alpha, beta, gamma:
        Strategic gains.  See module docstring.
    rng:
        Seeded :class:`numpy.random.Generator` — never call ``np.random``
        directly so determinism is preserved end-to-end.
    sigma_floor, sigma_naive:
        Forwarded to :func:`confabulation_width`.

    Returns
    -------
    tuple
        ``(p_star, sigma_conf, anchor)`` — the sampled claim, the kernel
        width, and the deterministic anchor used to centre the kernel.
    """
    anchor = strategic_anchor(trigger, prior_mean, mu_g, r_g, beta)
    sigma = confabulation_width(alpha, gamma, sigma_floor, sigma_naive)
    p_star = anchor + rng.normal(loc=0.0, scale=sigma, size=anchor.shape)
    return p_star, sigma, anchor


def pp_divergence(
    p_star: NDArray[np.float64],
    sigma_conf: float,
    anchor: NDArray[np.float64],
    bayes_mean: NDArray[np.float64],
    bayes_var: float,
) -> float:
    """Log Bayes factor between the Strategic (PP) and Bayesian models.

    .. math::
        \\Delta_{PP}(p^*) =
        \\log \\mathcal{N}\\bigl(p^*;\\,\\text{anchor},\\,\\sigma_{conf}^2 I\\bigr)
      - \\log \\mathcal{N}\\bigl(p^*;\\,\\mu_{post},\\,\\sigma_{post}^2 I\\bigr).

    A *positive* value means the observed claim is more likely under the
    Strategic generative model than under the Bayesian one — i.e. the agent
    is exhibiting Phantom Pinpoint.  The Bayesian negative-control ablation
    (A6, ``use_bayesian_baseline=True``) generates :math:`p^*` from the
    Bayesian model and therefore yields :math:`\\Delta_{PP} < 0` in
    expectation, which is exactly the discriminative signature we want.

    Parameters
    ----------
    p_star:
        ``(d,)`` claimed point.
    sigma_conf:
        Strategic claim kernel width :math:`\\sigma_{conf}`.
    anchor:
        ``(d,)`` strategic claim centre :math:`(1-\\beta)\\Pi_G(E)+\\beta\\mu`.
    bayes_mean, bayes_var:
        Parameters of the *post-trigger* Bayesian posterior.

    Returns
    -------
    float
        Log Bayes factor in nats.  Positive ⇒ PP signature.
    """
    p_star = np.asarray(p_star, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    bayes_mean = np.asarray(bayes_mean, dtype=np.float64)
    d = p_star.shape[-1]
    var_pp = max(sigma_conf**2, EPS)
    var_b = max(bayes_var, EPS)
    diff_pp_sq = float(np.sum((p_star - anchor) ** 2))
    diff_b_sq = float(np.sum((p_star - bayes_mean) ** 2))
    log_pp = -0.5 * d * float(np.log(2 * np.pi * var_pp)) - diff_pp_sq / (2 * var_pp)
    log_b = -0.5 * d * float(np.log(2 * np.pi * var_b)) - diff_b_sq / (2 * var_b)
    return log_pp - log_b


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Container for a single condition's simulation output.

    Attributes
    ----------
    p_star:
        ``(n_runs, d)`` claimed points.
    sigma_conf:
        ``(n_runs,)`` claim kernel widths.
    delta_pp:
        ``(n_runs,)`` per-run :math:`\\Delta_{PP}` values.
    post_hoc_fit:
        ``(n_runs,)`` boolean whether ``p* ∈ G`` and ``||p* − E|| < tau``.
    triggers:
        ``(n_runs, d)`` trigger locations used.
    bayes_means:
        ``(n_runs, d)`` Bayesian posterior means (counterfactual).
    metadata:
        Free-form dictionary of parameters used to generate the run.
    """

    p_star: NDArray[np.float64]
    sigma_conf: NDArray[np.float64]
    delta_pp: NDArray[np.float64]
    post_hoc_fit: NDArray[np.bool_]
    triggers: NDArray[np.float64]
    bayes_means: NDArray[np.float64]
    metadata: dict[str, float | int | str | bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PhantomPinpointModel:
    """Parameter bundle for a single homogeneous PP agent population.

    All fields are intentionally scalar / small so that an entire condition
    fits in one dataclass; population-level heterogeneity is introduced via
    the ``alpha_dist`` / ``beta_dist`` / ``gamma_dist`` shape parameters.

    Parameters
    ----------
    dim:
        Dimensionality of :math:`\\Theta`.  ``2`` is the canonical setting
        (axis 1 = "intent strength", axis 2 = "preferred timing"), allowing
        figures to remain interpretable.
    mu_g:
        ``(dim,)`` centre of :math:`G`.
    r_g:
        Hard radius :math:`r_g` defining membership in :math:`G`.
    g_sigma:
        Soft Gaussian standard deviation used to *generate* triggers that
        graze the goal region with controlled probability.
    tau:
        Likelihood width :math:`\\tau` (sensor / observation noise).
    prior_var:
        :math:`\\sigma_\\psi^2`.  Vague prior → larger PP signal.
    alpha_dist, beta_dist, gamma_dist:
        ``(a, b)`` Beta-distribution shape parameters.  ``gamma`` is also
        scaled by ``audience_size`` at sample time.
    audience_size:
        Multiplicative scalar on ``gamma`` — 0 = anonymous condition.
    sigma_floor, sigma_naive:
        Forwarded to :func:`confabulation_width`.
    use_bayesian_baseline:
        If ``True``, agents bypass the strategic claim and report the
        Bayesian posterior mean (ablation A6 — negative control).
    shuffle_g:
        If ``True``, randomise ``mu_g`` per run (ablation A4 — target shuffle).
    shuffle_e:
        If ``True``, replace the trigger by an i.i.d. uniform draw (ablation
        A5 — temporal shuffle).
    """

    dim: int = 2
    mu_g: tuple[float, ...] = (0.0, 0.0)
    r_g: float = 1.0
    g_sigma: float = 1.5
    tau: float = 0.3
    prior_var: float = 4.0
    alpha_dist: tuple[float, float] = (2.0, 2.0)
    beta_dist: tuple[float, float] = (2.0, 5.0)
    gamma_dist: tuple[float, float] = (2.0, 2.0)
    audience_size: float = 1.0
    sigma_floor: float = 0.02
    sigma_naive: float = 0.5
    use_bayesian_baseline: bool = False
    shuffle_g: bool = False
    shuffle_e: bool = False

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim!r}")
        if len(self.mu_g) != self.dim:
            raise ValueError(
                f"mu_g length {len(self.mu_g)} != dim {self.dim}"
            )
        if self.r_g <= 0 or self.g_sigma <= 0 or self.tau <= 0:
            raise ValueError("r_g, g_sigma, tau must be > 0")
        if self.prior_var <= 0:
            raise ValueError("prior_var must be > 0")
        if self.audience_size < 0:
            raise ValueError("audience_size must be >= 0")
        for name, dist in (
            ("alpha_dist", self.alpha_dist),
            ("beta_dist", self.beta_dist),
            ("gamma_dist", self.gamma_dist),
        ):
            if len(dist) != 2 or dist[0] <= 0 or dist[1] <= 0:
                raise ValueError(f"{name} must be (a>0, b>0), got {dist!r}")

    # --------------------------------------------------------------------- #
    # Forward simulation                                                    #
    # --------------------------------------------------------------------- #
    def simulate(self, n_runs: int, seed: int = 42) -> SimulationResult:
        """Run ``n_runs`` independent agents through one trigger event each.

        Parameters
        ----------
        n_runs:
            Number of i.i.d. agents.
        seed:
            Seed for :func:`numpy.random.default_rng`.

        Returns
        -------
        SimulationResult
            Vectorised per-agent quantities, ready for bootstrap CI analysis.
        """
        if n_runs <= 0:
            raise ValueError(f"n_runs must be > 0, got {n_runs!r}")
        rng = np.random.default_rng(seed)

        mu_g = np.asarray(self.mu_g, dtype=np.float64)
        d = self.dim

        # Per-agent traits (Beta-distributed in [0, 1]).
        alphas = rng.beta(*self.alpha_dist, size=n_runs)
        betas = rng.beta(*self.beta_dist, size=n_runs)
        gammas = rng.beta(*self.gamma_dist, size=n_runs) * self.audience_size

        # Per-agent prior mean — vague Gaussian centred near goal.
        prior_means = mu_g + rng.normal(
            0.0, np.sqrt(self.prior_var), size=(n_runs, d)
        )

        # Trigger generator — Gaussian around mu_g with std g_sigma so that
        # roughly half the triggers graze G when r_g ≈ g_sigma.
        triggers = mu_g + rng.normal(0.0, self.g_sigma, size=(n_runs, d))

        # Optional ablation shuffles.
        if self.shuffle_g:
            mu_g_per = mu_g + rng.normal(0.0, self.g_sigma, size=(n_runs, d))
        else:
            mu_g_per = np.broadcast_to(mu_g, (n_runs, d))

        if self.shuffle_e:
            triggers = rng.uniform(
                low=mu_g - 4 * self.g_sigma,
                high=mu_g + 4 * self.g_sigma,
                size=(n_runs, d),
            )

        p_star = np.empty((n_runs, d), dtype=np.float64)
        sigma_conf = np.empty(n_runs, dtype=np.float64)
        delta_pp = np.empty(n_runs, dtype=np.float64)
        post_hoc_fit = np.empty(n_runs, dtype=np.bool_)
        bayes_means = np.empty((n_runs, d), dtype=np.float64)

        for i in range(n_runs):
            mean_post, var_post = bayesian_posterior_mean(
                prior_means[i], self.prior_var, triggers[i], self.tau**2
            )
            bayes_means[i] = mean_post

            anchor = strategic_anchor(
                triggers[i], prior_means[i], mu_g_per[i], self.r_g, float(betas[i]),
            )
            sigma = confabulation_width(
                float(alphas[i]),
                float(gammas[i]),
                self.sigma_floor,
                self.sigma_naive,
            )
            sigma_conf[i] = sigma

            if self.use_bayesian_baseline:
                # Negative control — claim drawn from Bayesian posterior
                # (centre = bayes_mean) but with the *same* sampling width as
                # the strategic agent would use, so any difference in ΔPP is
                # attributable to *location*, not *width*.
                p_star[i] = mean_post + rng.normal(0.0, sigma, size=d)
            else:
                p_star[i] = anchor + rng.normal(0.0, sigma, size=d)

            delta_pp[i] = pp_divergence(
                p_star[i], sigma, anchor, bayes_means[i], var_post,
            )
            # ``post_hoc_fit`` is evaluated against the *baseline* G — even
            # under shuffle_g (per-agent random G) we test "did the claim
            # land in the population's true target".  This is the ablation
            # that breaks shuffles: claims drift into per-agent Gs that no
            # longer match the population G.
            in_g = bool(np.linalg.norm(p_star[i] - mu_g) <= self.r_g)
            close_e = bool(np.linalg.norm(p_star[i] - triggers[i]) < self.tau * 3)
            post_hoc_fit[i] = in_g and close_e

        _LOG.debug(
            "simulate done n=%d cond=%s mean Δ_PP=%.4f",
            n_runs,
            "bayes" if self.use_bayesian_baseline else "strategic",
            float(delta_pp.mean()),
        )
        return SimulationResult(
            p_star=p_star,
            sigma_conf=sigma_conf,
            delta_pp=delta_pp,
            post_hoc_fit=post_hoc_fit,
            triggers=triggers,
            bayes_means=bayes_means,
            metadata={
                "n_runs": int(n_runs),
                "seed": int(seed),
                "use_bayesian_baseline": bool(self.use_bayesian_baseline),
                "shuffle_g": bool(self.shuffle_g),
                "shuffle_e": bool(self.shuffle_e),
                "audience_size": float(self.audience_size),
                "r_g": float(self.r_g),
            },
        )
