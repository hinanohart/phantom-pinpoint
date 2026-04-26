"""Experiment 05 — repeated-trigger sensitivity.

Pre-registered hypothesis H6
----------------------------
Holding all other parameters fixed, drawing the trigger from a sharper
Gaussian (smaller ``g_sigma``) — i.e. *frequent* grazing of G — increases the
mean ΔPP because more agents experience an in-G trigger.  Specifically the
Spearman correlation between ``g_sigma`` and mean ΔPP is negative
(:math:`\\rho < -0.5`, two-sided p < 0.05).
"""

from __future__ import annotations

import sys

import pandas as pd
from scipy import stats

from experiments._common import (
    FIGURES_DIR,
    ensure_dirs,
    write_manifest,
    write_parquet,
)
from phantom_pinpoint.ablations import BASELINE_KWARGS
from phantom_pinpoint.simulation import run_condition
from phantom_pinpoint.statistics import bootstrap_ci
from phantom_pinpoint.visualization import plot_claim_geometry, save_figure

G_SIGMAS = (0.5, 1.0, 1.5, 2.5, 4.0)


def main(n_runs: int = 600, seed: int = 42) -> int:
    ensure_dirs()
    frames = []
    summary = []
    for i, gs in enumerate(G_SIGMAS):
        kwargs = {**BASELINE_KWARGS, "g_sigma": gs}
        df = run_condition(f"g_sigma={gs}", n_runs=n_runs, seed=seed + i, **kwargs)
        frames.append(df)
        ci = bootstrap_ci(df["delta_pp"].to_numpy(), seed=seed + i)
        summary.append(
            {
                "g_sigma": gs,
                "delta_mean": ci.statistic,
                "delta_ci_lo": ci.ci_lo,
                "delta_ci_hi": ci.ci_hi,
                "fit_rate": float(df["post_hoc_fit"].mean()),
            }
        )
    df_all = pd.concat(frames, ignore_index=True)
    write_parquet(df_all, "05_repeated_trigger")

    rho, p_value = stats.spearmanr(
        [s["g_sigma"] for s in summary], [s["delta_mean"] for s in summary]
    )

    fig = plot_claim_geometry(df_all, condition=f"g_sigma={G_SIGMAS[0]}")
    save_figure(fig, FIGURES_DIR / "fig04_geometry.pdf")
    save_figure(plot_claim_geometry(df_all, condition=f"g_sigma={G_SIGMAS[0]}"),
                FIGURES_DIR / "fig04_geometry.png")

    write_manifest(
        "05_repeated_trigger",
        {
            "n_runs": n_runs,
            "seed": seed,
            "g_sigmas": list(G_SIGMAS),
            "summary": summary,
            "spearman_g_sigma_delta": {"rho": float(rho), "p": float(p_value)},
            "H6_passed": bool(rho < -0.5 and p_value < 0.05),
        },
    )
    print(f"Spearman ΔPP vs g_sigma: ρ={rho:.3f} p={p_value:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
