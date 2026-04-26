"""Statistical primitives required by the pre-registered analysis plan.

CRITICAL RULE R5 forbids point-estimate Δ-claims; every test reported in the
paper / README must come with a bootstrap 95 % confidence interval, a
two-sided p-value, and a standard error.  This module provides:

* :func:`bootstrap_ci` — a thin, deterministic wrapper over
  :func:`scipy.stats.bootstrap` that also returns SE and a p-value against
  ``H0: statistic = 0`` via the percentile method.
* :func:`permutation_test` — exact-when-possible permutation test for the
  difference of means, used to compare Strategic vs Bayesian conditions.
* :func:`benjamini_hochberg` — BH-FDR correction with q = 0.05 default,
  applied across the H1–H5 family of pre-registered hypotheses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from phantom_pinpoint._logging import get_logger

_LOG = get_logger("statistics")

DEFAULT_RESAMPLES: Final[int] = 10_000
DEFAULT_CI: Final[float] = 0.95


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Bundle returned by :func:`bootstrap_ci`.

    Attributes
    ----------
    statistic:
        Point estimate of the chosen statistic on the observed data.
    ci_lo, ci_hi:
        Lower / upper bounds of the 95 % (or chosen) confidence interval.
    standard_error:
        Bootstrap standard error.
    p_two_sided:
        p-value against ``H0: statistic = 0`` based on the bootstrap
        distribution.  Computed as ``2 * min(P(B >= 0), P(B <= 0))``.
    n:
        Number of observations used.
    n_resamples:
        Number of bootstrap resamples drawn.
    """

    statistic: float
    ci_lo: float
    ci_hi: float
    standard_error: float
    p_two_sided: float
    n: int
    n_resamples: int

    @property
    def is_significant(self) -> bool:
        """``True`` iff the CI does not contain zero."""
        return (self.ci_lo > 0.0) or (self.ci_hi < 0.0)

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "statistic": self.statistic,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "standard_error": self.standard_error,
            "p_two_sided": self.p_two_sided,
            "n": self.n,
            "n_resamples": self.n_resamples,
            "is_significant": self.is_significant,
        }


def bootstrap_ci(
    data: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float] = np.mean,
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 42,
) -> BootstrapResult:
    """Compute a bootstrap CI, SE and p-value for ``statistic(data)``.

    Parameters
    ----------
    data:
        ``(n,)`` 1-D float array.
    statistic:
        Callable taking ``ndarray -> float``.  Defaults to the sample mean.
    n_resamples:
        Bootstrap sample count.  10 000 is the lower bound recommended by
        Efron & Tibshirani 1993 for percentile CIs at the 95 % level.
    ci:
        Confidence level in (0, 1).
    seed:
        Seed for ``numpy.random.default_rng``.  Determinism is mandatory for
        the reproducibility acceptance criterion AC6.

    Returns
    -------
    BootstrapResult
        See class docstring.

    Raises
    ------
    ValueError
        If ``data`` is empty, ``ci`` is out of range, or ``n_resamples < 100``.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    if data.size == 0:
        raise ValueError("bootstrap_ci received empty data")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be >= 100, got {n_resamples!r}")

    rng = np.random.default_rng(seed)
    res = stats.bootstrap(
        (data,),
        statistic,
        n_resamples=n_resamples,
        confidence_level=ci,
        method="percentile",
        random_state=rng,
        vectorized=False,
    )
    point = float(statistic(data))
    boot = np.asarray(res.bootstrap_distribution, dtype=np.float64)
    p_ge = float((boot >= 0).mean())
    p_le = float((boot <= 0).mean())
    p_two = float(min(1.0, 2.0 * min(p_ge, p_le)))
    se = float(boot.std(ddof=1))
    return BootstrapResult(
        statistic=point,
        ci_lo=float(res.confidence_interval.low),
        ci_hi=float(res.confidence_interval.high),
        standard_error=se,
        p_two_sided=p_two,
        n=int(data.size),
        n_resamples=int(n_resamples),
    )


def permutation_test(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    statistic: Callable[
        [NDArray[np.float64], NDArray[np.float64]], float
    ] = lambda x, y: float(np.mean(x) - np.mean(y)),
    n_resamples: int = DEFAULT_RESAMPLES,
    alternative: str = "two-sided",
    seed: int = 42,
) -> tuple[float, float]:
    """Permutation test for ``statistic(a, b)``.

    Wraps :func:`scipy.stats.permutation_test` with the ``"independent"``
    permutation type — observations are pooled and reshuffled — which is the
    correct null for "Strategic vs Bayesian" group-difference comparisons.

    Returns
    -------
    tuple
        ``(observed_statistic, p_value)``.
    """
    rng = np.random.default_rng(seed)
    res = stats.permutation_test(
        (np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)),
        statistic,
        n_resamples=n_resamples,
        alternative=alternative,
        permutation_type="independent",
        random_state=rng,
        vectorized=False,
    )
    return float(res.statistic), float(res.pvalue)


def benjamini_hochberg(p_values: NDArray[np.float64], q: float = 0.05) -> NDArray[np.bool_]:
    """Vanilla Benjamini–Hochberg multiple-testing correction.

    Parameters
    ----------
    p_values:
        ``(m,)`` array of two-sided p-values.
    q:
        False discovery rate.

    Returns
    -------
    numpy.ndarray
        Boolean array of length ``m`` flagging hypotheses *rejected* under
        BH-FDR control at level ``q``.
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size == 0:
        return np.zeros(0, dtype=np.bool_)
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0, 1), got {q!r}")
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    if not below.any():
        return np.zeros(m, dtype=np.bool_)
    k = int(np.max(np.where(below)[0]))
    rejected_sorted = np.zeros(m, dtype=np.bool_)
    rejected_sorted[: k + 1] = True
    rejected = np.empty(m, dtype=np.bool_)
    rejected[order] = rejected_sorted
    return rejected
