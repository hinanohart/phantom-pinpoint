"""Experiment 01 — baseline characterisation of the Phantom Pinpoint signature.

Pre-registered hypothesis H1
----------------------------
Under the baseline condition (audience γ ~ Beta(2, 2) ⋅ size 1, vague prior
:math:`\\sigma_\\psi^2 = 4`, sharp likelihood :math:`\\tau = 0.3`), the
log-Bayes-factor ΔPP has a 95 % bootstrap CI strictly above zero (n = 500).

Acceptance criterion AC1: ΔPP CI lower bound > 0.
"""

from __future__ import annotations

import sys

import pandas as pd

from experiments._common import (
    FIGURES_DIR,
    ensure_dirs,
    write_manifest,
    write_parquet,
)
from phantom_pinpoint.ablations import BASELINE_KWARGS
from phantom_pinpoint.simulation import run_condition
from phantom_pinpoint.statistics import bootstrap_ci
from phantom_pinpoint.visualization import plot_specificity_collapse, save_figure


def main(n_runs: int = 800, seed: int = 42) -> int:
    ensure_dirs()
    df_strategic = run_condition("baseline_strategic", n_runs=n_runs, seed=seed, **BASELINE_KWARGS)
    df_bayesian = run_condition(
        "baseline_bayesian", n_runs=n_runs, seed=seed,
        **{**BASELINE_KWARGS, "use_bayesian_baseline": True},
    )
    df = pd.concat([df_strategic, df_bayesian], ignore_index=True)

    write_parquet(df, "01_baseline")

    res_strat = bootstrap_ci(df_strategic["delta_pp"].to_numpy(), seed=seed)
    res_bayes = bootstrap_ci(df_bayesian["delta_pp"].to_numpy(), seed=seed + 1)

    fig = plot_specificity_collapse(df)
    save_figure(fig, FIGURES_DIR / "fig01_baseline_violin.pdf")
    save_figure(plot_specificity_collapse(df), FIGURES_DIR / "fig01_baseline_violin.png")

    write_manifest(
        "01_baseline",
        {
            "n_runs": n_runs,
            "seed": seed,
            "baseline_strategic": res_strat.as_dict(),
            "baseline_bayesian": res_bayes.as_dict(),
            "AC1_passed": bool(res_strat.ci_lo > 0.0),
            "AC2_passed": bool(res_bayes.ci_hi < 0.0),
        },
    )
    print(f"H1 strategic: {res_strat.as_dict()}")
    print(f"H1 bayesian neg-control: {res_bayes.as_dict()}")
    return 0 if (res_strat.ci_lo > 0 and res_bayes.ci_hi < 0) else 1


if __name__ == "__main__":
    sys.exit(main())
