"""High-level helpers to run named experimental conditions.

This module shapes the per-condition output into tidy ``pandas`` DataFrames so
that downstream analysis (statistics, ablations, visualisation) is uniform.

Each row corresponds to *one agent* and carries:

* ``condition`` (str)  — name of the condition
* ``run_id`` (int)     — within-condition sequential id
* ``delta_pp`` (float) — primary endpoint
* ``sigma_conf`` (float)
* ``post_hoc_fit`` (bool)
* ``trigger_x`` / ``trigger_y`` (float, only the first 2 dims for tidiness)
* ``p_star_x`` / ``p_star_y`` (float)
* ``audience_size`` (float)
* ``shuffle_g`` / ``shuffle_e`` (bool)
* ``use_bayesian_baseline`` (bool)
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from phantom_pinpoint._logging import get_logger
from phantom_pinpoint.core import PhantomPinpointModel, SimulationResult

_LOG = get_logger("simulation")


def _to_dataframe(name: str, res: SimulationResult, model: PhantomPinpointModel) -> pd.DataFrame:
    n = res.delta_pp.shape[0]
    d = res.p_star.shape[1]
    df = pd.DataFrame(
        {
            "condition": np.repeat(name, n),
            "run_id": np.arange(n, dtype=np.int64),
            "delta_pp": res.delta_pp,
            "sigma_conf": res.sigma_conf,
            "post_hoc_fit": res.post_hoc_fit.astype(bool),
            "trigger_x": res.triggers[:, 0],
            "trigger_y": res.triggers[:, 1] if d > 1 else 0.0,
            "p_star_x": res.p_star[:, 0],
            "p_star_y": res.p_star[:, 1] if d > 1 else 0.0,
            "audience_size": np.full(n, model.audience_size, dtype=np.float64),
            "shuffle_g": np.full(n, model.shuffle_g, dtype=bool),
            "shuffle_e": np.full(n, model.shuffle_e, dtype=bool),
            "use_bayesian_baseline": np.full(n, model.use_bayesian_baseline, dtype=bool),
            "r_g": np.full(n, model.r_g, dtype=np.float64),
        }
    )
    return df


def run_condition(
    name: str,
    n_runs: int = 500,
    seed: int = 42,
    **model_kwargs: Any,
) -> pd.DataFrame:
    """Run one named condition and return a tidy DataFrame.

    Parameters
    ----------
    name:
        Free-form condition identifier — usually one of ``"baseline"``,
        ``"no_audience"``, etc.
    n_runs:
        Population size.  500 is the lower bound mandated by the
        pre-registration; experiments raise it as needed for power.
    seed:
        Seed forwarded to :meth:`PhantomPinpointModel.simulate`.
    model_kwargs:
        Forwarded directly to :class:`PhantomPinpointModel`.

    Returns
    -------
    pandas.DataFrame
        Tidy long-format with one row per agent.
    """
    model = PhantomPinpointModel(**model_kwargs)
    res = model.simulate(n_runs=n_runs, seed=seed)
    df = _to_dataframe(name, res, model)
    _LOG.info(
        "condition=%s n=%d mean Δ_PP=%.4f post-hoc fit rate=%.3f",
        name,
        n_runs,
        float(df["delta_pp"].mean()),
        float(df["post_hoc_fit"].mean()),
    )
    return df


def run_grid(
    conditions: dict[str, dict[str, Any]],
    n_runs: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a dictionary of conditions and concatenate the results.

    Parameters
    ----------
    conditions:
        ``{condition_name: {model kwargs}}``.
    n_runs, seed:
        Forwarded to :func:`run_condition` for every condition.

    Returns
    -------
    pandas.DataFrame
        Concatenation of all per-condition DataFrames.
    """
    frames = [
        run_condition(name=name, n_runs=n_runs, seed=seed, **kwargs)
        for name, kwargs in conditions.items()
    ]
    return pd.concat(frames, ignore_index=True)


def parameter_sweep(
    base_kwargs: dict[str, Any],
    sweep: dict[str, Iterable[Any]],
    n_runs: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Cartesian-product sweep over selected model fields.

    Convenient for sensitivity analyses (acceptance criterion AC7).

    Parameters
    ----------
    base_kwargs:
        Default :class:`PhantomPinpointModel` kwargs.
    sweep:
        Mapping ``param -> iterable of values``.
    n_runs, seed:
        Forwarded.

    Returns
    -------
    pandas.DataFrame
        Combined output, with sweep parameters appearing as additional columns
        labelled ``swp__<param>``.
    """
    keys = list(sweep.keys())
    grids = list(product(*[list(sweep[k]) for k in keys]))
    frames: list[pd.DataFrame] = []
    for combo in grids:
        kwargs = dict(base_kwargs)
        kwargs.update(dict(zip(keys, combo, strict=True)))
        name = "_".join(f"{k}={v}" for k, v in zip(keys, combo, strict=True))
        df = run_condition(name, n_runs=n_runs, seed=seed, **kwargs)
        for k, v in zip(keys, combo, strict=True):
            df[f"swp__{k}"] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
