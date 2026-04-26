"""Experiment 02 — audience-driven specificity sharpening.

Pre-registered hypothesis H2
----------------------------
Across audience sizes :math:`\\gamma \\in \\{0, 0.5, 1, 2, 4\\}`, the mean
claim kernel width :math:`\\sigma_{conf}` decreases monotonically (Spearman
:math:`\\rho < -0.5`, two-sided p < 0.01) and ΔPP increases monotonically
(Spearman :math:`\\rho > +0.5`, two-sided p < 0.01).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from phantom_pinpoint.ablations import BASELINE_KWARGS
from phantom_pinpoint.simulation import run_condition
from phantom_pinpoint.statistics import bootstrap_ci
from phantom_pinpoint.visualization import plot_audience_response, save_figure

from experiments._common import (
    FIGURES_DIR,
    ensure_dirs,
    write_manifest,
    write_parquet,
)

AUDIENCE_LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0)


def main(n_runs: int = 600, seed: int = 42) -> int:
    ensure_dirs()
    frames = []
    summary = []
    for i, audience in enumerate(AUDIENCE_LEVELS):
        kwargs = {**BASELINE_KWARGS, "audience_size": audience}
        df = run_condition(f"audience_{audience}", n_runs=n_runs, seed=seed + i, **kwargs)
        frames.append(df)
        ci_sigma = bootstrap_ci(df["sigma_conf"].to_numpy(), seed=seed + i)
        ci_delta = bootstrap_ci(df["delta_pp"].to_numpy(), seed=seed + i + 100)
        summary.append(
            {
                "audience_size": audience,
                "sigma_mean": ci_sigma.statistic,
                "sigma_ci_lo": ci_sigma.ci_lo,
                "sigma_ci_hi": ci_sigma.ci_hi,
                "delta_mean": ci_delta.statistic,
                "delta_ci_lo": ci_delta.ci_lo,
                "delta_ci_hi": ci_delta.ci_hi,
            }
        )
    df_all = pd.concat(frames, ignore_index=True)
    write_parquet(df_all, "02_audience")

    sigmas = np.array([s["sigma_mean"] for s in summary])
    deltas = np.array([s["delta_mean"] for s in summary])
    rho_sigma, p_sigma = stats.spearmanr(AUDIENCE_LEVELS, sigmas)
    rho_delta, p_delta = stats.spearmanr(AUDIENCE_LEVELS, deltas)

    fig = plot_audience_response(df_all)
    save_figure(fig, FIGURES_DIR / "fig02_audience.pdf")
    save_figure(plot_audience_response(df_all), FIGURES_DIR / "fig02_audience.png")

    write_manifest(
        "02_audience",
        {
            "n_runs": n_runs,
            "seed": seed,
            "audience_levels": list(AUDIENCE_LEVELS),
            "summary": summary,
            "spearman_sigma": {"rho": float(rho_sigma), "p": float(p_sigma)},
            "spearman_delta": {"rho": float(rho_delta), "p": float(p_delta)},
            "H2_sigma_monotone_passed": bool(rho_sigma < -0.5 and p_sigma < 0.01),
            "H2_delta_monotone_passed": bool(rho_delta > 0.5 and p_delta < 0.01),
        },
    )
    print(f"Spearman σ_conf vs γ: ρ={rho_sigma:.3f} p={p_sigma:.3g}")
    print(f"Spearman ΔPP    vs γ: ρ={rho_delta:.3f} p={p_delta:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
