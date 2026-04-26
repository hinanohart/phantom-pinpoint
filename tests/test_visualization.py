"""Smoke tests for the visualisation layer.

Figure correctness is hard to unit-test; we instead verify that each
``plot_*`` factory returns a non-empty :class:`matplotlib.figure.Figure`
when given a representative input, and that ``save_figure`` produces a
non-zero-byte file at the requested path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt  # noqa: E402

from phantom_pinpoint.simulation import run_condition  # noqa: E402
from phantom_pinpoint.visualization import (  # noqa: E402
    plot_ablation_heatmap,
    plot_audience_response,
    plot_claim_geometry,
    plot_specificity_collapse,
    save_figure,
)


def _df():
    a = run_condition("a", n_runs=64, seed=1, audience_size=0.0)
    b = run_condition("b", n_runs=64, seed=2, audience_size=2.0)
    import pandas as pd

    return pd.concat([a, b], ignore_index=True)


def test_plot_specificity_collapse() -> None:
    fig = plot_specificity_collapse(_df())
    assert isinstance(fig, plt.Figure)
    assert fig.axes
    plt.close(fig)


def test_plot_audience_response() -> None:
    fig = plot_audience_response(_df())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_ablation_heatmap() -> None:
    df = _df().assign(ablation=lambda d: d["condition"])
    fig = plot_ablation_heatmap(df)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_claim_geometry() -> None:
    fig = plot_claim_geometry(_df(), condition="a")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_figure_atomic(tmp_path: Path) -> None:
    fig = plot_specificity_collapse(_df())
    out = tmp_path / "out.png"
    save_figure(fig, out)
    assert out.exists()
    assert out.stat().st_size > 0
    # Partial sibling must be cleaned up by atomic rename.
    assert not (tmp_path / ".out.partial.png").exists()
