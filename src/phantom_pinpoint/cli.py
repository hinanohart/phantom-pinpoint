"""Typer-based command-line entry point.

Usage::

    phantom-pinpoint baseline --n-runs 500
    phantom-pinpoint ablations --n-runs 1000 --output results/ablations.parquet
    phantom-pinpoint figure 01 --input results/baseline.parquet --output figures/fig1.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from phantom_pinpoint import __version__
from phantom_pinpoint._logging import get_logger
from phantom_pinpoint.ablations import BASELINE_KWARGS, ablation_grid
from phantom_pinpoint.simulation import run_condition
from phantom_pinpoint.statistics import bootstrap_ci
from phantom_pinpoint.visualization import (
    plot_ablation_heatmap,
    plot_audience_response,
    plot_claim_geometry,
    plot_specificity_collapse,
    save_figure,
)

app = typer.Typer(help="Phantom Pinpoint — simulator CLI")
_LOG = get_logger("cli")


@app.command()
def baseline(
    n_runs: int = 500,
    seed: int = 42,
    output: Path = Path("results/baseline.parquet"),
) -> None:
    """Run the baseline condition and persist a Parquet file."""
    df = run_condition("baseline", n_runs=n_runs, seed=seed, **BASELINE_KWARGS)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output)
    res = bootstrap_ci(df["delta_pp"].to_numpy())
    typer.echo(json.dumps(res.as_dict(), indent=2))


@app.command()
def ablations(
    n_runs: int = 1000,
    seed: int = 42,
    output: Path = Path("results/ablations.parquet"),
) -> None:
    """Run all pre-registered ablations and persist a Parquet file."""
    df = ablation_grid(n_runs=n_runs, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output)
    typer.echo(f"wrote {output} ({len(df)} rows)")


@app.command()
def figure(
    name: str,
    input_file: Path = typer.Option(..., "--input", "-i"),
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    """Regenerate a named figure from a results Parquet file.

    name must be one of: ``specificity``, ``audience``, ``ablation``, ``geometry``.
    """
    df = pd.read_parquet(input_file)
    fig_factories = {
        "specificity": plot_specificity_collapse,
        "audience": plot_audience_response,
        "ablation": plot_ablation_heatmap,
        "geometry": plot_claim_geometry,
    }
    if name not in fig_factories:
        raise typer.BadParameter(f"unknown figure '{name}'.  Pick one of {list(fig_factories)}")
    fig = fig_factories[name](df)
    save_figure(fig, output)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
