"""Identifiability degeneracy diagnostic (AC9).

When :math:`E \\in G` *and* :math:`\\|\\mu - \\mu_G\\|` is small relative to
:math:`r_G`, the strategic anchor and the Bayesian posterior mean collapse
onto the *same* point, making the two competing generative models
**unidentifiable** for that draw.  Critic's AC9 demands that:

1. The simulator be able to *flag* such draws with a boolean diagnostic.
2. Aggregate statistics report what fraction of agents fall in the
   degenerate region.
3. The :math:`\\Delta_{PP}` distribution conditional on
   ``is_degenerate=True`` exhibit a 95 % bootstrap CI that *contains* zero,
   sealing the discriminative claim that PP only fires *outside* the
   degeneracy region.

The full ABC-SMC parameter recovery study is **deferred to v0.3.0**; this
module ships the lightweight diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from phantom_pinpoint._logging import get_logger
from phantom_pinpoint.statistics import bootstrap_ci

_LOG = get_logger("identifiability")

#: Default proximity ratio :math:`\\epsilon` such that
#: ``||mu - mu_g|| < eps * r_g`` flags the prior as "near goal".
DEFAULT_PRIOR_PROXIMITY: Final[float] = 0.1


@dataclass(frozen=True, slots=True)
class DegeneracyReport:
    """Aggregate report of identifiability degeneracy across a population.

    Attributes
    ----------
    fraction_degenerate:
        Share of agents flagged as degenerate.
    delta_pp_inside:
        Bootstrap CI for the mean :math:`\\Delta_{PP}` of degenerate draws.
        Should *contain* zero for AC9 to pass.
    delta_pp_outside:
        Bootstrap CI for non-degenerate draws.  Should *exclude* zero.
    n_inside, n_outside:
        Population sizes per region.
    ac9_passed:
        Boolean — both region CIs satisfy the discriminative criterion.
    """

    fraction_degenerate: float
    delta_pp_inside_lo: float
    delta_pp_inside_hi: float
    delta_pp_outside_lo: float
    delta_pp_outside_hi: float
    n_inside: int
    n_outside: int
    ac9_passed: bool


def detect_degeneracy(
    triggers: NDArray[np.float64],
    prior_means: NDArray[np.float64],
    mu_g: NDArray[np.float64],
    r_g: float,
    *,
    prior_proximity: float = DEFAULT_PRIOR_PROXIMITY,
) -> NDArray[np.bool_]:
    """Element-wise mask of agents in the degeneracy region.

    An agent is *degenerate* when **both**:

    * The trigger is inside the goal region: :math:`\\|E - \\mu_G\\| \\le r_G`.
    * The prior mean is close to the goal centre:
      :math:`\\|\\mu - \\mu_G\\| < \\epsilon\\, r_G`.

    Parameters
    ----------
    triggers:
        ``(n, d)`` per-agent trigger locations.
    prior_means:
        ``(n, d)`` per-agent prior means.
    mu_g, r_g:
        Goal centre and radius.
    prior_proximity:
        :math:`\\epsilon` in :math:`\\|\\mu - \\mu_G\\| < \\epsilon\\, r_G`.

    Returns
    -------
    numpy.ndarray of bool, shape ``(n,)``
    """
    triggers = np.asarray(triggers, dtype=np.float64)
    prior_means = np.asarray(prior_means, dtype=np.float64)
    mu_g = np.asarray(mu_g, dtype=np.float64)
    if triggers.shape != prior_means.shape:
        raise ValueError(
            f"shape mismatch: triggers {triggers.shape}, priors {prior_means.shape}"
        )
    if r_g <= 0:
        raise ValueError("r_g must be > 0")
    if prior_proximity <= 0:
        raise ValueError("prior_proximity must be > 0")
    trigger_in_g = np.linalg.norm(triggers - mu_g, axis=-1) <= r_g
    prior_near_g = np.linalg.norm(prior_means - mu_g, axis=-1) < prior_proximity * r_g
    return np.asarray(trigger_in_g & prior_near_g, dtype=np.bool_)


def assess(
    delta_pp: NDArray[np.float64],
    is_degenerate: NDArray[np.bool_],
    *,
    seed: int = 42,
    n_resamples: int = 5_000,
) -> DegeneracyReport:
    """Compute the AC9 report for a population.

    Parameters
    ----------
    delta_pp:
        ``(n,)`` per-agent log Bayes factor.
    is_degenerate:
        ``(n,)`` boolean mask from :func:`detect_degeneracy`.
    seed:
        Bootstrap seed.
    n_resamples:
        Bootstrap iterations per region.

    Returns
    -------
    DegeneracyReport
    """
    delta_pp = np.asarray(delta_pp, dtype=np.float64).ravel()
    is_degenerate = np.asarray(is_degenerate, dtype=np.bool_).ravel()
    if delta_pp.shape != is_degenerate.shape:
        raise ValueError("shape mismatch")
    inside = delta_pp[is_degenerate]
    outside = delta_pp[~is_degenerate]
    n_in, n_out = int(inside.size), int(outside.size)
    if n_in < 5 or n_out < 5:
        _LOG.warning(
            "degeneracy assessment underpowered: n_inside=%d n_outside=%d",
            n_in, n_out,
        )

    if n_in >= 5:
        ci_in = bootstrap_ci(inside, n_resamples=n_resamples, seed=seed)
        in_lo, in_hi = ci_in.ci_lo, ci_in.ci_hi
        in_contains_zero = in_lo <= 0 <= in_hi
    else:
        in_lo = in_hi = float("nan")
        in_contains_zero = False
    if n_out >= 5:
        ci_out = bootstrap_ci(outside, n_resamples=n_resamples, seed=seed + 1)
        out_lo, out_hi = ci_out.ci_lo, ci_out.ci_hi
        out_excludes_zero = (out_lo > 0) or (out_hi < 0)
    else:
        out_lo = out_hi = float("nan")
        out_excludes_zero = False

    return DegeneracyReport(
        fraction_degenerate=float(is_degenerate.mean()),
        delta_pp_inside_lo=float(in_lo),
        delta_pp_inside_hi=float(in_hi),
        delta_pp_outside_lo=float(out_lo),
        delta_pp_outside_hi=float(out_hi),
        n_inside=n_in,
        n_outside=n_out,
        ac9_passed=bool(in_contains_zero and out_excludes_zero),
    )


def detect_from_dataframe(
    df: pd.DataFrame,
    mu_g: NDArray[np.float64],
    r_g: float,
    prior_means: NDArray[np.float64],
    *,
    prior_proximity: float = DEFAULT_PRIOR_PROXIMITY,
) -> pd.DataFrame:
    """DataFrame helper — adds an ``is_degenerate`` column.

    Convenience wrapper used by :mod:`experiments.06_sensitivity_sweep` and
    :mod:`experiments.09_location_decomposition`.
    """
    triggers = df[["trigger_x", "trigger_y"]].to_numpy(dtype=np.float64)
    mask = detect_degeneracy(
        triggers, prior_means, mu_g, r_g, prior_proximity=prior_proximity,
    )
    out = df.copy()
    out["is_degenerate"] = mask
    return out
