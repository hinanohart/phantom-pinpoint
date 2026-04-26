"""Sensitivity sweep and parameter elasticity (AC10 — formerly AC7).

The v0.1.0 release deferred the sensitivity sweep to v0.2.0; this module
provides the implementation.  Two interfaces:

* :func:`sensitivity_sweep` — one-axis-at-a-time univariate sweep with
  bootstrap CIs at every grid point.  Cheapest, used as the v0.2.0
  acceptance gate (``AC10``).
* :func:`elasticity` — log–log slope of the response variable wrt each
  swept parameter.  Reports a single dimensionless number per axis.

Sobol global sensitivity indices are intentionally **deferred to v0.3.0**
because (a) they require :math:`n \\sim 10^4` per axis combination and
(b) the v0.2.0 acceptance criterion is *robustness of sign*, not a
quantitative variance decomposition.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from phantom_pinpoint._logging import get_logger
from phantom_pinpoint.simulation import run_condition
from phantom_pinpoint.statistics import bootstrap_ci

_LOG = get_logger("sensitivity")


@dataclass(frozen=True, slots=True)
class SensitivityCell:
    """One cell of a sensitivity sweep.

    Attributes
    ----------
    axis:
        Name of the parameter being swept (e.g. ``"r_g"``).
    value:
        Value taken by the parameter at this grid point.
    metric:
        Name of the response variable (e.g. ``"delta_pp"``).
    ci_lo, ci_hi, statistic:
        Bootstrap CI for the *mean* of the response over ``n_runs`` agents.
    """

    axis: str
    value: float
    metric: str
    statistic: float
    ci_lo: float
    ci_hi: float


def sensitivity_sweep(
    base_kwargs: dict[str, Any],
    axis: str,
    values: Iterable[float],
    *,
    metric: str = "delta_pp",
    n_runs: int = 500,
    seed: int = 42,
    n_resamples: int = 5_000,
) -> pd.DataFrame:
    """Univariate sweep: vary ``axis`` over ``values``, hold others fixed.

    Parameters
    ----------
    base_kwargs:
        Default :class:`PhantomPinpointModel` kwargs (typically
        ``BASELINE_KWARGS``).
    axis:
        Name of the model field to sweep.
    values:
        Iterable of values to try.
    metric:
        Name of the simulation column to summarise.
    n_runs:
        Population size per grid point.
    seed:
        Master seed; cell ``i`` uses ``seed + i``.
    n_resamples:
        Bootstrap CI iterations per cell.

    Returns
    -------
    pandas.DataFrame
        One row per cell with the standard simulation columns plus
        ``ci_lo``, ``ci_hi``, ``statistic``.
    """
    if axis not in {*base_kwargs.keys(), "audience_size", "r_g", "g_sigma", "tau", "prior_var"}:
        # `axis` may be a valid kwarg even if not in base_kwargs
        _LOG.warning("axis %r is not in base_kwargs — assuming valid model field", axis)

    rows: list[SensitivityCell] = []
    for i, v in enumerate(values):
        kwargs = dict(base_kwargs)
        kwargs[axis] = v
        df = run_condition(f"{axis}={v}", n_runs=n_runs, seed=seed + i, **kwargs)
        if metric not in df.columns:
            raise KeyError(
                f"metric {metric!r} not found in DataFrame columns {list(df.columns)}"
            )
        data = df[metric].to_numpy(dtype=np.float64)
        ci = bootstrap_ci(data, n_resamples=n_resamples, seed=seed + i)
        rows.append(
            SensitivityCell(
                axis=axis,
                value=float(v),
                metric=metric,
                statistic=ci.statistic,
                ci_lo=ci.ci_lo,
                ci_hi=ci.ci_hi,
            )
        )
    return pd.DataFrame([asdict(r) for r in rows])


def elasticity(frame: pd.DataFrame, *, target: str = "statistic") -> float:
    """Log–log slope ``d log(target) / d log(value)`` over the sweep.

    Returns ``np.nan`` if any value is ≤ 0 or the response crosses zero
    (log of negative is undefined).  Useful for AC10 robustness reporting:
    an elasticity with absolute value > 0.5 marks a "high-sensitivity" axis.
    """
    vals = frame["value"].to_numpy(dtype=np.float64)
    targ = frame[target].to_numpy(dtype=np.float64)
    if (vals <= 0).any() or (targ <= 0).any():
        return float("nan")
    log_v = np.log(vals)
    log_t = np.log(targ)
    slope, _ = np.polyfit(log_v, log_t, 1)
    return float(slope)


def robustness_check(
    base_kwargs: dict[str, Any],
    axes: dict[str, Iterable[float]],
    *,
    metric: str = "delta_pp",
    n_runs: int = 500,
    seed: int = 42,
    sign_threshold: float = 0.0,
) -> pd.DataFrame:
    """AC10 robustness gate: does the *sign* of ``metric`` survive sweeping?

    For each axis runs :func:`sensitivity_sweep` and records:

    * ``sign_preserved``: fraction of cells whose CI excludes ``sign_threshold``
      on the *correct* side relative to the median cell.
    * ``min_statistic``, ``max_statistic``: range across cells.

    AC10 passes for an axis iff ``sign_preserved >= 0.9``.

    Returns
    -------
    pandas.DataFrame
        One row per axis.
    """
    rows = []
    for i, (axis, values) in enumerate(axes.items()):
        sweep = sensitivity_sweep(
            base_kwargs, axis, values,
            metric=metric, n_runs=n_runs, seed=seed + 100 * i,
        )
        median_stat = float(np.median(sweep["statistic"]))
        sign = +1 if median_stat > sign_threshold else -1
        if sign > 0:
            preserved = (sweep["ci_lo"] > sign_threshold).mean()
        else:
            preserved = (sweep["ci_hi"] < sign_threshold).mean()
        rows.append(
            {
                "axis": axis,
                "n_cells": len(sweep),
                "sign_preserved": float(preserved),
                "min_statistic": float(sweep["statistic"].min()),
                "max_statistic": float(sweep["statistic"].max()),
                "ac10_passed": bool(preserved >= 0.9),
            }
        )
    return pd.DataFrame(rows)
