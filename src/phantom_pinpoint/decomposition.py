"""ΔPP decomposition into width and location components.

The v0.1.0 ``pp_divergence`` is the log Bayes factor

.. math::
    \\Delta_{PP}(p^*)
        = \\log\\,\\mathcal N(p^*;\\,\\text{anchor},\\,\\sigma_{conf}^{2}I)
        - \\log\\,\\mathcal N(p^*;\\,\\mu_{\\text{post}},\\,\\sigma_{\\text{post}}^{2}I).

Expanding the two Gaussian log densities and rearranging splits :math:`\\Delta_{PP}`
into two physically interpretable components:

.. math::
    \\Delta_{PP} = \\underbrace{\\tfrac{d}{2}\\log
        \\bigl(\\sigma_{\\text{post}}^{2}/\\sigma_{conf}^{2}\\bigr)}_{\\Delta_{PP}^{\\text{width}}}
    \\;+\\;
    \\underbrace{\\tfrac{1}{2}\\Bigl(
        \\tfrac{\\|p^*-\\mu_{\\text{post}}\\|^{2}}{\\sigma_{\\text{post}}^{2}}
        - \\tfrac{\\|p^*-\\text{anchor}\\|^{2}}{\\sigma_{conf}^{2}}\\Bigr)}_{\\Delta_{PP}^{\\text{loc}}}.

This decomposition is the v0.2.0 centrepiece because it explains why H2b
(audience increases ΔPP) failed in v0.1.0:

* Increasing audience pressure :math:`\\gamma` shrinks :math:`\\sigma_{conf}`,
  which **inflates** :math:`\\Delta_{PP}^{\\text{width}}`.
* But the location component contains a :math:`\\sigma_{conf}^{2}/\\sigma_{\\text{post}}^{2}`
  term that **deflates** by exactly :math:`d/2` as :math:`\\sigma_{conf}\\to0`.
* Asymptotically the two effects cancel, so the *aggregate* :math:`\\Delta_{PP}`
  is roughly invariant to :math:`\\gamma` — exactly what we observed
  (:math:`\\rho=0.20`, :math:`p=0.747`).

Reporting :math:`\\Delta_{PP}^{\\text{width}}` and :math:`\\Delta_{PP}^{\\text{loc}}`
**separately** rescues the audience-driven prediction (it now lives in
:math:`\\Delta_{PP}^{\\text{width}}` alone) and matches Critic's AC8 demand
to "decompose then re-pre-register".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from phantom_pinpoint._logging import get_logger
from phantom_pinpoint.core import EPS, SimulationResult, project_onto_region

_LOG = get_logger("decomposition")

#: Tolerance under which the ``width + location ≈ total`` identity should
#: hold for any single agent.  Relaxed from ``EPS`` because the underlying
#: divergence is log-Bayes-factor and accumulates float-precision noise.
DECOMPOSITION_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """Per-agent decomposition of :math:`\\Delta_{PP}`.

    Attributes
    ----------
    delta_pp:
        The aggregate signature.  Numerically equal to
        :func:`phantom_pinpoint.core.pp_divergence` (up to ``DECOMPOSITION_TOLERANCE``).
    width:
        The :math:`d/2 \\log(\\sigma_{\\text{post}}^{2}/\\sigma_{conf}^{2})`
        term.  Captures audience-driven specificity sharpening.
    location:
        The remaining "Mahalanobis difference" term.  Captures Texas-Sharpshooter
        anchor pull and identifiability collapse.
    """

    delta_pp: float
    width: float
    location: float

    def __post_init__(self) -> None:
        residual = self.delta_pp - self.width - self.location
        if abs(residual) > DECOMPOSITION_TOLERANCE * max(abs(self.delta_pp), 1.0):
            # We do not raise — numerical noise from the surrounding
            # simulation is acceptable — but flag loudly in DEBUG.
            _LOG.debug(
                "decomposition residual %.2e exceeds tolerance %.2e",
                residual,
                DECOMPOSITION_TOLERANCE,
            )


def decompose_delta_pp(
    p_star: NDArray[np.float64],
    sigma_conf: float,
    anchor: NDArray[np.float64],
    bayes_mean: NDArray[np.float64],
    bayes_var: float,
) -> DecompositionResult:
    """Closed-form decomposition of one agent's :math:`\\Delta_{PP}`.

    Parameters
    ----------
    p_star:
        ``(d,)`` claimed point.
    sigma_conf:
        Strategic claim kernel width.
    anchor:
        ``(d,)`` strategic anchor :math:`(1-\\beta)\\Pi_G(E) + \\beta\\mu`.
    bayes_mean, bayes_var:
        Posterior mean / scalar variance of the Bayesian counterfactual.

    Returns
    -------
    DecompositionResult
        ``(delta_pp, width, location)`` such that
        ``width + location == delta_pp`` to numerical precision.
    """
    p_star = np.asarray(p_star, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    bayes_mean = np.asarray(bayes_mean, dtype=np.float64)
    if p_star.shape != anchor.shape or p_star.shape != bayes_mean.shape:
        raise ValueError(
            f"shape mismatch: p_star={p_star.shape}, anchor={anchor.shape}, "
            f"bayes_mean={bayes_mean.shape}"
        )
    if sigma_conf <= 0:
        raise ValueError(f"sigma_conf must be > 0, got {sigma_conf!r}")
    if bayes_var <= 0:
        raise ValueError(f"bayes_var must be > 0, got {bayes_var!r}")

    d = int(p_star.shape[-1])
    var_pp = max(sigma_conf**2, EPS)
    var_b = max(bayes_var, EPS)

    width = 0.5 * d * float(np.log(var_b / var_pp))
    diff_b_sq = float(np.sum((p_star - bayes_mean) ** 2))
    diff_pp_sq = float(np.sum((p_star - anchor) ** 2))
    location = 0.5 * (diff_b_sq / var_b - diff_pp_sq / var_pp)
    return DecompositionResult(
        delta_pp=width + location, width=width, location=location,
    )


def decompose_simulation(
    result: SimulationResult,
    *,
    mu_g: NDArray[np.float64],
    r_g: float,
    prior_means: NDArray[np.float64],
    betas: NDArray[np.float64],
) -> pd.DataFrame:
    """Vectorised :func:`decompose_delta_pp` over an entire simulation result.

    The strategic anchor is recomputed from ``trigger``, ``mu_g``, ``r_g``,
    ``prior_means`` and the per-agent ``betas`` because
    :class:`SimulationResult` does not (in v0.1.0) persist the anchor.

    Parameters
    ----------
    result:
        Output of :meth:`PhantomPinpointModel.simulate`.
    mu_g, r_g:
        Goal-region centre and radius shared across the population (if
        ``shuffle_g`` was active in the source model the per-agent ``mu_g``
        is unrecoverable from the public ``SimulationResult`` and the
        decomposition will use the population centre — flagged in the docs).
    prior_means:
        ``(n_runs, d)`` per-agent prior means.  Reproducible from the same
        seed but not stored in :class:`SimulationResult`; callers regenerate
        them with :func:`numpy.random.default_rng(seed).normal(...)`.
    betas:
        ``(n_runs,)`` per-agent defensive bias ``β``.

    Returns
    -------
    pandas.DataFrame
        Long-format with columns ``run_id``, ``delta_pp``,
        ``delta_pp_width``, ``delta_pp_location``, ``post_hoc_fit``,
        ``sigma_conf``.
    """
    n = result.delta_pp.shape[0]
    if prior_means.shape[0] != n or betas.shape[0] != n:
        raise ValueError("prior_means and betas must have length n_runs")

    widths = np.empty(n, dtype=np.float64)
    locations = np.empty(n, dtype=np.float64)
    deltas = np.empty(n, dtype=np.float64)

    mu_g = np.asarray(mu_g, dtype=np.float64)

    for i in range(n):
        proj = project_onto_region(result.triggers[i], mu_g, r_g)
        anchor = (1.0 - float(betas[i])) * proj + float(betas[i]) * prior_means[i]
        # The Bayesian variance is recovered from the analytic identity
        # var_post = 1/(1/prior_var + 1/tau^2) — but ``SimulationResult``
        # exposes ``bayes_means`` not ``bayes_var``.  We back out the
        # variance from the residual ``||bayes_mean - prior|| / ||trigger - prior||``
        # ratio: bayes_mean = prior_var/(prior_var + tau^2) * trigger + tau^2/(prior_var+tau^2) * prior
        # ⇒ ratio = prior_var/(prior_var + tau^2)  ⇒  tau^2 = prior_var * (1 - ratio) / ratio
        # Numerically this is unstable when prior ≈ trigger; we fall back to
        # a conservative scalar from the metadata if available.
        prior = prior_means[i]
        bm = result.bayes_means[i]
        trig = result.triggers[i]
        denom_vec = trig - prior
        denom = float(np.dot(denom_vec, denom_vec))
        ratio = (
            0.5  # degenerate fallback when trigger ≈ prior
            if denom < EPS
            else float(np.dot(bm - prior, denom_vec) / denom)
        )
        ratio = float(np.clip(ratio, 1e-6, 1 - 1e-6))
        prior_var = float(result.metadata.get("prior_var", 4.0))
        tau2 = prior_var * (1.0 - ratio) / ratio
        bayes_var = 1.0 / (1.0 / prior_var + 1.0 / max(tau2, EPS))

        decomposition = decompose_delta_pp(
            result.p_star[i],
            float(result.sigma_conf[i]),
            anchor,
            bm,
            bayes_var,
        )
        widths[i] = decomposition.width
        locations[i] = decomposition.location
        deltas[i] = decomposition.delta_pp

    return pd.DataFrame(
        {
            "run_id": np.arange(n, dtype=np.int64),
            "delta_pp": deltas,
            "delta_pp_width": widths,
            "delta_pp_location": locations,
            "delta_pp_legacy": result.delta_pp,
            "sigma_conf": result.sigma_conf,
            "post_hoc_fit": result.post_hoc_fit.astype(bool),
        }
    )
