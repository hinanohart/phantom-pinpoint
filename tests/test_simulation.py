"""End-to-end tests for the simulation orchestration layer."""

from __future__ import annotations

import pandas as pd

from phantom_pinpoint.simulation import parameter_sweep, run_condition, run_grid


def test_run_condition_columns() -> None:
    df = run_condition("baseline", n_runs=64, seed=42)
    expected = {
        "condition", "run_id", "delta_pp", "sigma_conf", "post_hoc_fit",
        "trigger_x", "trigger_y", "p_star_x", "p_star_y",
        "audience_size", "shuffle_g", "shuffle_e", "use_bayesian_baseline", "r_g",
    }
    assert expected.issubset(df.columns)
    assert len(df) == 64
    assert df["condition"].nunique() == 1


def test_run_grid_concatenates() -> None:
    df = run_grid(
        {
            "baseline": {"audience_size": 1.0},
            "no_audience": {"audience_size": 0.0},
        },
        n_runs=32,
    )
    assert isinstance(df, pd.DataFrame)
    assert df["condition"].nunique() == 2
    assert len(df) == 64


def test_parameter_sweep_records_swept_columns() -> None:
    df = parameter_sweep(
        base_kwargs={"audience_size": 1.0},
        sweep={"r_g": [0.5, 1.0]},
        n_runs=16,
    )
    assert "swp__r_g" in df.columns
    assert sorted(df["swp__r_g"].unique().tolist()) == [0.5, 1.0]


def test_audience_increases_specificity() -> None:
    """Pre-registered H1: larger audience → narrower claim kernel σ_conf."""
    private = run_condition("priv", n_runs=400, audience_size=0.0, seed=1)
    public = run_condition("pub", n_runs=400, audience_size=3.0, seed=1)
    assert public["sigma_conf"].mean() < private["sigma_conf"].mean()
