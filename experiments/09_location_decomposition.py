"""Experiment 09 — H2b/H6 root cause via ΔPP decomposition.

Pre-registered hypothesis H9 (location-driven audience effect)
--------------------------------------------------------------
Across audience sizes :math:`\\gamma \\in \\{0, 0.5, 1, 2, 4\\}`, the
*width* component :math:`\\Delta_{PP}^{\\text{width}}` increases
monotonically with :math:`\\gamma` (Spearman :math:`\\rho > +0.5`,
two-sided p < 0.01) **even though** the aggregate :math:`\\Delta_{PP}`
does not.  This rescues the v0.1.0 H2b failure as a decomposition
artifact.

Pre-registered hypothesis H10 (location explains H6 reversal)
-------------------------------------------------------------
Across :math:`g_\\sigma \\in \\{0.5, 1.0, 1.5, 2.5, 4.0\\}`, the *location*
component :math:`\\Delta_{PP}^{\\text{loc}}` is positively correlated with
:math:`g_\\sigma` (Spearman :math:`\\rho > +0.5`), confirming that wider
trigger distributions inflate the location term — the analytic explanation
of v0.1.0's H6 sign reversal.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from experiments._common import (
    FIGURES_DIR,
    ensure_dirs,
    write_manifest,
    write_parquet,
)
from phantom_pinpoint.ablations import BASELINE_KWARGS
from phantom_pinpoint.core import PhantomPinpointModel
from phantom_pinpoint.decomposition import decompose_simulation
from phantom_pinpoint.visualization import save_figure

AUDIENCE_LEVELS = (0.0, 0.5, 1.0, 2.0, 4.0)
G_SIGMA_LEVELS = (0.5, 1.0, 1.5, 2.5, 4.0)


def _decompose(model: PhantomPinpointModel, n_runs: int, seed: int) -> pd.DataFrame:
    res = model.simulate(n_runs=n_runs, seed=seed)
    rng = np.random.default_rng(seed)
    n = n_runs
    _alphas = rng.beta(*model.alpha_dist, size=n)
    betas = rng.beta(*model.beta_dist, size=n)
    _gammas = rng.beta(*model.gamma_dist, size=n) * model.audience_size
    prior_means = np.asarray(model.mu_g, dtype=np.float64) + rng.normal(
        0.0, np.sqrt(model.prior_var), size=(n, model.dim),
    )
    return decompose_simulation(
        res, mu_g=np.asarray(model.mu_g), r_g=model.r_g,
        prior_means=prior_means, betas=betas,
    )


def _plot_decomposition(df: pd.DataFrame, x: str, label: str) -> plt.Figure:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    melt = df.melt(
        id_vars=[x],
        value_vars=["delta_pp_width", "delta_pp_location", "delta_pp"],
        var_name="component", value_name="nats",
    )
    sns.lineplot(
        data=melt, x=x, y="nats", hue="component",
        estimator="mean", errorbar=("ci", 95),
        markers=True, dashes=False, ax=ax,
    )
    ax.axhline(0.0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel(f"{label}")
    ax.set_ylabel(r"$\Delta_{PP}$ component (nats)")
    ax.set_title(f"ΔPP decomposition vs {label}")
    fig.tight_layout()
    return fig


def main(n_runs: int = 500, seed: int = 42) -> int:
    ensure_dirs()

    # H9 — audience sweep
    audience_frames = []
    for i, audience in enumerate(AUDIENCE_LEVELS):
        kwargs = {**BASELINE_KWARGS, "audience_size": audience}
        model = PhantomPinpointModel(**kwargs)
        df = _decompose(model, n_runs, seed + 17 * i).assign(audience_size=audience)
        audience_frames.append(df)
    aud = pd.concat(audience_frames, ignore_index=True)
    write_parquet(aud, "09_audience_decomposition")

    # H10 — g_sigma sweep
    gs_frames = []
    for i, gs in enumerate(G_SIGMA_LEVELS):
        kwargs = {**BASELINE_KWARGS, "g_sigma": gs}
        model = PhantomPinpointModel(**kwargs)
        df = _decompose(model, n_runs, seed + 31 * i).assign(g_sigma=gs)
        gs_frames.append(df)
    gs_df = pd.concat(gs_frames, ignore_index=True)
    write_parquet(gs_df, "09_gsigma_decomposition")

    # Aggregate stats
    aud_means = aud.groupby("audience_size").agg(
        width_mean=("delta_pp_width", "mean"),
        loc_mean=("delta_pp_location", "mean"),
        total_mean=("delta_pp", "mean"),
    ).reset_index()
    rho_w, p_w = stats.spearmanr(
        aud_means["audience_size"], aud_means["width_mean"],
    )
    rho_total, p_total = stats.spearmanr(
        aud_means["audience_size"], aud_means["total_mean"],
    )

    gs_means = gs_df.groupby("g_sigma").agg(
        width_mean=("delta_pp_width", "mean"),
        loc_mean=("delta_pp_location", "mean"),
        total_mean=("delta_pp", "mean"),
    ).reset_index()
    rho_loc_gs, p_loc_gs = stats.spearmanr(
        gs_means["g_sigma"], gs_means["loc_mean"],
    )

    # Figures
    save_figure(
        _plot_decomposition(aud, "audience_size", r"audience size $\gamma$"),
        FIGURES_DIR / "fig09_audience_decomposition.pdf",
    )
    save_figure(
        _plot_decomposition(aud, "audience_size", r"audience size $\gamma$"),
        FIGURES_DIR / "fig09_audience_decomposition.png",
    )
    save_figure(
        _plot_decomposition(gs_df, "g_sigma", r"$g_\sigma$"),
        FIGURES_DIR / "fig09_gsigma_decomposition.pdf",
    )
    save_figure(
        _plot_decomposition(gs_df, "g_sigma", r"$g_\sigma$"),
        FIGURES_DIR / "fig09_gsigma_decomposition.png",
    )

    write_manifest(
        "09_location_decomposition",
        {
            "n_runs": n_runs,
            "seed": seed,
            "audience_means": aud_means.to_dict(orient="records"),
            "gsigma_means": gs_means.to_dict(orient="records"),
            "spearman_width_vs_audience": {"rho": float(rho_w), "p": float(p_w)},
            "spearman_total_vs_audience": {"rho": float(rho_total), "p": float(p_total)},
            "spearman_loc_vs_gsigma": {"rho": float(rho_loc_gs), "p": float(p_loc_gs)},
            "H9_passed": bool(rho_w > 0.5 and p_w < 0.01),
            "H10_passed": bool(rho_loc_gs > 0.5 and p_loc_gs < 0.01),
        },
    )
    print(f"Spearman ΔPP_width vs γ: ρ={rho_w:.3f} p={p_w:.3g}")
    print(f"Spearman ΔPP_total vs γ: ρ={rho_total:.3f} p={p_total:.3g}")
    print(f"Spearman ΔPP_loc vs g_σ: ρ={rho_loc_gs:.3f} p={p_loc_gs:.3g}")
    print(aud_means.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
