"""Effect size estimators with bootstrap confidence intervals.

R5 (no point-estimate Δ-claims) explicitly bans naked Spearman ρ values
because, with small *n* of conditions (e.g. 5 audience levels), a perfect
:math:`\\rho = +1.0` carries virtually no statistical evidence.  Critic's
AC12 therefore mandates that every contrast in the v0.2.0 paper be
accompanied by a *standardised* effect size with a 95 % bootstrap CI.

This module provides:

* :func:`cohens_d` — Cohen's *d* (independent-samples or paired)
* :func:`hedges_g` — bias-corrected Hedges' *g*
* :func:`cliffs_delta` — non-parametric, distribution-free
* :func:`bootstrap_effect_size` — generic bootstrap CI wrapper

All estimators are pure numpy/scipy and reproduce bit-exactly given a
seeded :class:`numpy.random.Generator`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from phantom_pinpoint._logging import get_logger

_LOG = get_logger("effect_size")

DEFAULT_RESAMPLES: Final[int] = 10_000
DEFAULT_CI: Final[float] = 0.95


@dataclass(frozen=True, slots=True)
class EffectSizeCI:
    """Bundle of an effect-size point estimate with bootstrap CI.

    Attributes
    ----------
    name:
        Short identifier (``"cohens_d"``, ``"hedges_g"``, ``"cliffs_delta"``).
    statistic:
        Point estimate computed on the observed data.
    ci_lo, ci_hi:
        Lower / upper bounds of the chosen-level bootstrap CI.
    n_a, n_b:
        Sample sizes of the two groups.
    n_resamples:
        Number of bootstrap resamples drawn.
    """

    name: str
    statistic: float
    ci_lo: float
    ci_hi: float
    n_a: int
    n_b: int
    n_resamples: int

    @property
    def is_significant(self) -> bool:
        """``True`` iff the bootstrap CI excludes zero."""
        return (self.ci_lo > 0.0) or (self.ci_hi < 0.0)

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "name": self.name,
            "statistic": self.statistic,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_resamples": self.n_resamples,
            "is_significant": self.is_significant,
        }


def cohens_d(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    paired: bool = False,
) -> float:
    """Cohen's *d* effect size.

    For independent samples, the pooled-SD definition (Hedges 1981)::

        d = (mean(a) - mean(b)) / sqrt(((n_a-1) s_a^2 + (n_b-1) s_b^2) / (n_a+n_b-2))

    For paired samples::

        d = mean(a - b) / std(a - b, ddof=1)

    Parameters
    ----------
    a, b:
        Two 1-D float arrays of equal length (paired) or arbitrary length
        (independent).
    paired:
        If ``True`` use the difference-score definition.  Defaults to ``False``.

    Returns
    -------
    float
        Cohen's *d*.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("inputs must be non-empty")
    if paired:
        if a.size != b.size:
            raise ValueError("paired Cohen's d requires equal-length arrays")
        diff = a - b
        s = float(diff.std(ddof=1))
        if s <= 0:
            return 0.0 if float(diff.mean()) == 0.0 else float("inf")
        return float(diff.mean() / s)
    n_a, n_b = a.size, b.size
    s_a, s_b = float(a.std(ddof=1)), float(b.std(ddof=1))
    pooled = float(np.sqrt(((n_a - 1) * s_a**2 + (n_b - 1) * s_b**2) / (n_a + n_b - 2)))
    if pooled <= 0:
        return 0.0 if float(a.mean()) == float(b.mean()) else float("inf")
    return float((a.mean() - b.mean()) / pooled)


def hedges_g(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    paired: bool = False,
) -> float:
    """Bias-corrected Hedges' *g*.

    Multiplies Cohen's *d* by the small-sample correction factor

    .. math::
        J = 1 - \\frac{3}{4(n_a+n_b)-9},

    valid for the independent-samples case.  For paired the correction
    uses :math:`4n - 5` in the denominator (Cumming 2012).
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = cohens_d(a, b, paired=paired)
    if paired:
        n = a.size
        if n <= 2:
            return d
        j = 1.0 - 3.0 / (4 * n - 5)
    else:
        n = a.size + b.size
        if n <= 3:
            return d
        j = 1.0 - 3.0 / (4 * n - 9)
    return float(d * j)


def cliffs_delta(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Cliff's :math:`\\delta` non-parametric effect size.

    Defined as :math:`P(X_a > X_b) - P(X_b > X_a)` over independent draws
    from the empirical distributions; ranges in [-1, +1].  Computed via the
    Mann–Whitney U statistic for *O(n log n)* runtime instead of the naive
    *O(n_a · n_b)*.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError("inputs must be non-empty")
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    n_a, n_b = a.size, b.size
    return float(2.0 * u / (n_a * n_b) - 1.0)


def bootstrap_effect_size(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64], NDArray[np.float64]], float] = cohens_d,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 42,
    name: str | None = None,
) -> EffectSizeCI:
    """Bootstrap CI for *any* two-sample effect size.

    Resamples each group **with replacement** independently
    (independent-samples bootstrap).  For paired data, callers should
    pre-difference and use :func:`bootstrap_paired_mean` instead.

    Parameters
    ----------
    a, b:
        Two 1-D float arrays.
    statistic:
        Callable ``(a, b) -> float`` (default :func:`cohens_d`).
    n_resamples:
        Bootstrap iterations (default 10 000).
    ci:
        Confidence level in (0, 1).
    seed:
        RNG seed for determinism.
    name:
        Optional label stored in the returned :class:`EffectSizeCI`.

    Returns
    -------
    EffectSizeCI
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        raise ValueError("each group needs >=2 samples for bootstrap")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be >= 100, got {n_resamples!r}")

    rng = np.random.default_rng(seed)
    point = float(statistic(a, b))
    boot = np.empty(n_resamples, dtype=np.float64)
    n_a, n_b = a.size, b.size
    for i in range(n_resamples):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        boot[i] = float(statistic(a[idx_a], b[idx_b]))
    alpha = 1.0 - ci
    lo, hi = float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))
    label = name if name is not None else getattr(statistic, "__name__", "effect_size")
    return EffectSizeCI(
        name=label,
        statistic=point,
        ci_lo=lo,
        ci_hi=hi,
        n_a=int(n_a),
        n_b=int(n_b),
        n_resamples=int(n_resamples),
    )
