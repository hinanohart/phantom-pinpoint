"""Experiment 04 — pre-registered ablation grid (A1 – A7).

Hypotheses
----------
* AC2 — A6 (Bayesian baseline) yields a ΔPP 95 % CI strictly *below* zero
  (negative control).
* H4 — A1 (α≈0) and A3 (γ=0) reduce mean ΔPP by ≥ 50 % vs baseline.
* H5 — A4 (shuffle G) and A5 (shuffle E) drop the post-hoc-fit rate by
  ≥ 0.10 absolute vs baseline.

Multiple testing: H1–H5 are corrected with Benjamini–Hochberg q = 0.05.
"""

from __future__ import annotations

import sys

import numpy as np

from experiments._common import (
    FIGURES_DIR,
    ensure_dirs,
    write_manifest,
    write_parquet,
)
from phantom_pinpoint.ablations import ABLATIONS, ablation_grid
from phantom_pinpoint.statistics import (
    benjamini_hochberg,
    bootstrap_ci,
    permutation_test,
)
from phantom_pinpoint.visualization import plot_ablation_heatmap, save_figure


def main(n_runs: int = 1000, seed: int = 42) -> int:
    ensure_dirs()
    df = ablation_grid(n_runs=n_runs, seed=seed)
    write_parquet(df, "04_ablations")

    summary = []
    p_values = []
    baseline_delta = df.loc[df["ablation"] == "baseline", "delta_pp"].to_numpy()
    baseline_fit = df.loc[df["ablation"] == "baseline", "post_hoc_fit"].to_numpy().astype(float)

    for i, name in enumerate(ABLATIONS.keys()):
        sub = df[df["ablation"] == name]
        ci_delta = bootstrap_ci(sub["delta_pp"].to_numpy(), seed=seed + i)
        ci_fit = bootstrap_ci(
            sub["post_hoc_fit"].to_numpy().astype(float), seed=seed + i + 1000,
        )
        if name == "baseline":
            p_value = 1.0
            stat_diff = 0.0
        else:
            stat_diff, p_value = permutation_test(
                sub["delta_pp"].to_numpy(),
                baseline_delta,
                seed=seed + i + 2000,
                n_resamples=5_000,
            )
        summary.append(
            {
                "ablation": name,
                "delta": ci_delta.as_dict(),
                "fit": ci_fit.as_dict(),
                "perm_p_vs_baseline": float(p_value),
                "perm_stat_vs_baseline": float(stat_diff),
            }
        )
        p_values.append(float(p_value))

    bh = benjamini_hochberg(np.asarray(p_values), q=0.05)
    for row, rejected in zip(summary, bh, strict=True):
        row["bh_rejected"] = bool(rejected)

    a6 = next(s for s in summary if s["ablation"] == "A6_bayesian")
    a4 = next(s for s in summary if s["ablation"] == "A4_shuffleG")
    a5 = next(s for s in summary if s["ablation"] == "A5_shuffleE")

    fig = plot_ablation_heatmap(df)
    save_figure(fig, FIGURES_DIR / "fig03_ablations.pdf")
    save_figure(plot_ablation_heatmap(df), FIGURES_DIR / "fig03_ablations.png")

    write_manifest(
        "04_ablations",
        {
            "n_runs": n_runs,
            "seed": seed,
            "summary": summary,
            "AC2_neg_control_passed": bool(a6["delta"]["ci_hi"] < 0.0),
            "H5_shuffle_g_drop_fit": float(
                next(s for s in summary if s["ablation"] == "baseline")["fit"]["statistic"]
                - a4["fit"]["statistic"]
            ),
            "H5_shuffle_e_drop_fit": float(
                next(s for s in summary if s["ablation"] == "baseline")["fit"]["statistic"]
                - a5["fit"]["statistic"]
            ),
        },
    )
    for row in summary:
        print(
            f"{row['ablation']:>16}  "
            f"Δ={row['delta']['statistic']:+.3f}  "
            f"[{row['delta']['ci_lo']:+.2f}, {row['delta']['ci_hi']:+.2f}]  "
            f"fit={row['fit']['statistic']:.3f}  "
            f"p={row['perm_p_vs_baseline']:.3g}  bh={row['bh_rejected']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
