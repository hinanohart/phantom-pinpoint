"""Reproducible figures (Fig 1 – 4) for the Phantom Pinpoint paper.

All functions accept a tidy long DataFrame and return a ``matplotlib.Figure``.
Saving is deliberately *not* performed inside the plotting functions — the
experiment scripts choose the destination so that ``scripts/reproduce_all.sh``
can write atomically (tmp → rename) under a known directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from phantom_pinpoint._logging import get_logger

_LOG = get_logger("visualization")

_PALETTE = sns.color_palette("viridis", as_cmap=False)


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def plot_specificity_collapse(df: pd.DataFrame) -> plt.Figure:
    """Fig 1 — :math:`\\Delta_{PP}` distribution per condition (violin)."""
    _set_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    sns.violinplot(
        data=df,
        x="condition",
        y="delta_pp",
        ax=ax,
        inner="quartile",
        cut=0,
        density_norm="width",
        palette=_PALETTE,
        hue="condition",
        legend=False,
    )
    ax.axhline(0.0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel("Condition")
    ax.set_ylabel(r"$\Delta_{PP}$ (nats)")
    ax.set_title("Phantom-Pinpoint signature across conditions")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    return fig


def plot_audience_response(df: pd.DataFrame) -> plt.Figure:
    """Fig 2 — claim sharpness vs audience size (mean ± 95 % CI)."""
    _set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    grp = df.groupby("audience_size")["sigma_conf"].agg(["mean", "std", "count"])
    se = grp["std"] / np.sqrt(grp["count"])
    ci = 1.96 * se
    ax.errorbar(
        grp.index, grp["mean"], yerr=ci,
        marker="o", lw=1.6, capsize=3, color=_PALETTE[2],
    )
    ax.set_xlabel("Audience size $\\gamma$")
    ax.set_ylabel(r"Claim kernel width $\sigma_{conf}$ (smaller = sharper claim)")
    ax.set_title("Audience-driven post-hoc specificity sharpening")
    fig.tight_layout()
    return fig


def plot_ablation_heatmap(ablation_df: pd.DataFrame) -> plt.Figure:
    """Fig 3 — mean :math:`\\Delta_{PP}` per ablation with 95 % CI annotations."""
    _set_style()
    grp = (
        ablation_df.groupby("ablation")["delta_pp"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["se"] = grp["std"] / np.sqrt(grp["count"])
    grp["ci_lo"] = grp["mean"] - 1.96 * grp["se"]
    grp["ci_hi"] = grp["mean"] + 1.96 * grp["se"]
    grp = grp.sort_values("mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.barh(
        grp["ablation"], grp["mean"],
        xerr=[grp["mean"] - grp["ci_lo"], grp["ci_hi"] - grp["mean"]],
        color=_PALETTE[3], ecolor="k", capsize=3,
    )
    for bar, val in zip(bars, grp["mean"], strict=True):
        ax.text(
            bar.get_width() + (0.02 if val >= 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel(r"$\Delta_{PP}$ (nats, mean $\pm$ 95% CI)")
    ax.set_ylabel("Ablation")
    ax.set_title("Pre-registered ablations against the PP signature")
    fig.tight_layout()
    return fig


def plot_claim_geometry(df: pd.DataFrame, condition: str | None = None) -> plt.Figure:
    """Fig 4 — geometry of trigger vs claim points in :math:`\\mathbb{R}^2`.

    Visualises the Texas-Sharpshooter intuition: claims clump near triggers
    when :math:`\\Delta_{PP}` is high.
    """
    _set_style()
    if condition is not None:
        df = df[df["condition"] == condition].copy()
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    sub = df.sample(min(800, len(df)), random_state=7) if len(df) > 800 else df
    ax.scatter(
        sub["trigger_x"], sub["trigger_y"],
        s=8, alpha=0.4, label="trigger $E$", color=_PALETTE[1],
    )
    ax.scatter(
        sub["p_star_x"], sub["p_star_y"],
        s=8, alpha=0.4, label="claim $p^{*}$", color=_PALETTE[4],
    )
    for _, row in sub.iterrows():
        ax.plot(
            [row["trigger_x"], row["p_star_x"]],
            [row["trigger_y"], row["p_star_y"]],
            color="k", lw=0.2, alpha=0.15,
        )
    theta = np.linspace(0, 2 * np.pi, 256)
    r_g = float(df["r_g"].iloc[0])
    ax.plot(r_g * np.cos(theta), r_g * np.sin(theta), color="r", lw=1.0, label="$\\partial G$")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_aspect("equal")
    ax.set_title(f"Trigger → claim geometry{' — ' + condition if condition else ''}")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Atomic save — write to a sibling ``.partial`` then rename.

    matplotlib infers the format from the file extension, so the partial file
    must keep the original suffix; we use a leading dot prefix on the *stem*
    to make the partial uniquely identifiable while preserving the extension.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.partial{path.suffix}")
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    tmp.replace(path)
    plt.close(fig)
    _LOG.info("saved %s", path)
