"""Pre-registered ablation grid (A1 – A7).

Each ablation is encoded as a kwargs dictionary that *modifies* the
``baseline`` condition in a single, theoretically motivated way.  This module
provides a single :func:`ablation_grid` entry point that returns a tidy long
DataFrame keyed by ablation name, ready for BH-FDR correction.

Ablation specification (mirrors :doc:`docs/preregistration.md`):

============  ===========================================  =====================
ID            change                                       prediction
============  ===========================================  =====================
A1 alpha0     ``alpha_dist = (0.001, 100)``  → α≈0          ΔPP ↓ (less spec)
A2 beta0      ``beta_dist  = (0.001, 100)``  → β≈0          ΔPP ↑ (no defense)
A3 gamma0     ``audience_size = 0``          → γ=0          ΔPP ↓ (no audience)
A4 shuffleG   ``shuffle_g = True``                          ΔPP → 0 (target)
A5 shuffleE   ``shuffle_e = True``                          ΔPP → 0 (trigger)
A6 bayes      ``use_bayesian_baseline = True``              ΔPP → 0 (neg ctrl)
A7 sigma0     ``sigma_floor = sigma_naive = 1e-6``          ΔPP ↑ (deg. claim)
============  ===========================================  =====================
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from phantom_pinpoint._logging import get_logger
from phantom_pinpoint.simulation import run_condition

_LOG = get_logger("ablations")


# Conservative baseline used as the reference for every ablation.
BASELINE_KWARGS: dict[str, Any] = {
    "dim": 2,
    "mu_g": (0.0, 0.0),
    "r_g": 1.0,
    "g_sigma": 1.5,
    "tau": 0.3,
    "prior_var": 4.0,
    "alpha_dist": (2.0, 2.0),
    "beta_dist": (2.0, 5.0),
    "gamma_dist": (2.0, 2.0),
    "audience_size": 1.0,
    "sigma_floor": 0.02,
    "sigma_naive": 0.5,
    "use_bayesian_baseline": False,
    "shuffle_g": False,
    "shuffle_e": False,
}


ABLATIONS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "A1_alpha0": {"alpha_dist": (0.001, 100.0)},
    "A2_beta0": {"beta_dist": (0.001, 100.0)},
    "A3_gamma0": {"audience_size": 0.0},
    "A4_shuffleG": {"shuffle_g": True},
    "A5_shuffleE": {"shuffle_e": True},
    "A6_bayesian": {"use_bayesian_baseline": True},
    "A7_sigma0": {"sigma_floor": 1e-6, "sigma_naive": 1e-6},
}


def ablation_grid(
    n_runs: int = 1000,
    seed: int = 42,
    base_kwargs: dict[str, Any] | None = None,
    ablations: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Run all ablations and return the concatenated DataFrame.

    Parameters
    ----------
    n_runs:
        Population size per ablation.  Defaults to 1000 for AC1 power ≥ 0.8.
    seed:
        Master seed.  Each ablation receives ``seed + i`` so independence is
        guaranteed across ablations while remaining deterministic.
    base_kwargs:
        Override of :data:`BASELINE_KWARGS` (useful for sensitivity sweeps).
    ablations:
        Override of :data:`ABLATIONS` (useful for adding A8, A9 later).

    Returns
    -------
    pandas.DataFrame
        Long-format with the standard simulation columns plus an ``ablation``
        column.
    """
    base = dict(base_kwargs) if base_kwargs is not None else dict(BASELINE_KWARGS)
    abls = dict(ablations) if ablations is not None else dict(ABLATIONS)
    frames = []
    for i, (name, override) in enumerate(abls.items()):
        kwargs = dict(base)
        kwargs.update(override)
        df = run_condition(name=name, n_runs=n_runs, seed=seed + i, **kwargs)
        df["ablation"] = name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    _LOG.info("ablation grid run: %d ablations, %d total rows", len(abls), len(out))
    return out
