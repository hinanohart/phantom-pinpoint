"""Experiment 06 — sensitivity sweep (AC10, formerly AC7).

Pre-registered hypothesis H7
----------------------------
Across ±50 % univariate sweeps of ``r_g``, ``g_sigma``, ``tau`` and
``audience_size``, the *sign* of the baseline ΔPP (positive) is preserved
in at least 90 % of grid cells.  AC10 passes iff every axis attains
``sign_preserved >= 0.9``.
"""

from __future__ import annotations

import sys

import pandas as pd

from experiments._common import (
    ensure_dirs,
    write_manifest,
    write_parquet,
)
from phantom_pinpoint.ablations import BASELINE_KWARGS
from phantom_pinpoint.sensitivity import (
    elasticity,
    robustness_check,
    sensitivity_sweep,
)


def _pm50(value: float) -> list[float]:
    """5-point ±50 % grid around ``value``."""
    return [0.5 * value, 0.75 * value, value, 1.25 * value, 1.5 * value]


def main(n_runs: int = 400, seed: int = 42) -> int:
    ensure_dirs()
    base = dict(BASELINE_KWARGS)
    axes = {
        "r_g": _pm50(base["r_g"]),
        "g_sigma": _pm50(base["g_sigma"]),
        "tau": _pm50(base["tau"]),
        "audience_size": _pm50(max(base["audience_size"], 0.5)),
    }

    sweeps: list[pd.DataFrame] = []
    for i, (axis, values) in enumerate(axes.items()):
        df = sensitivity_sweep(
            base, axis, values,
            metric="delta_pp", n_runs=n_runs,
            seed=seed + 17 * i, n_resamples=2000,
        )
        sweeps.append(df)
    full = pd.concat(sweeps, ignore_index=True)
    write_parquet(full, "06_sensitivity")

    elasticities = {
        axis: elasticity(s, target="statistic")
        for axis, s in zip(axes.keys(), sweeps, strict=True)
    }
    robust = robustness_check(
        base, axes,
        metric="delta_pp", n_runs=n_runs, seed=seed + 1000,
    )
    write_parquet(robust, "06_sensitivity_robustness")

    ac10 = bool(robust["ac10_passed"].all())
    write_manifest(
        "06_sensitivity",
        {
            "n_runs": n_runs,
            "seed": seed,
            "axes": {k: list(v) for k, v in axes.items()},
            "elasticities": {k: float(v) for k, v in elasticities.items()},
            "robustness": robust.to_dict(orient="records"),
            "AC10_passed": ac10,
        },
    )
    print(robust.to_string(index=False))
    print(f"AC10 sensitivity gate: {'PASS' if ac10 else 'FAIL'}")
    return 0 if ac10 else 1


if __name__ == "__main__":
    sys.exit(main())
