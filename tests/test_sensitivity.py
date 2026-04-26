"""Tests for the sensitivity sweep helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phantom_pinpoint.ablations import BASELINE_KWARGS
from phantom_pinpoint.sensitivity import (
    SensitivityCell,
    elasticity,
    robustness_check,
    sensitivity_sweep,
)


def test_sweep_basic() -> None:
    df = sensitivity_sweep(
        BASELINE_KWARGS, axis="r_g", values=[0.5, 1.0, 2.0],
        metric="delta_pp", n_runs=64, seed=1, n_resamples=500,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert {"axis", "value", "statistic", "ci_lo", "ci_hi"}.issubset(df.columns)
    assert (df["axis"] == "r_g").all()


def test_sensitivity_cell_dataclass() -> None:
    cell = SensitivityCell(
        axis="r_g", value=1.0, metric="delta_pp",
        statistic=10.0, ci_lo=8.0, ci_hi=12.0,
    )
    assert cell.value == 1.0


def test_elasticity_finite_for_positive() -> None:
    df = pd.DataFrame({"value": [0.5, 1.0, 2.0], "statistic": [1.0, 2.0, 4.0]})
    e = elasticity(df, target="statistic")
    # Linear in log-log → slope = 1
    assert abs(e - 1.0) < 1e-6


def test_elasticity_nan_when_negative() -> None:
    df = pd.DataFrame({"value": [-1.0, 1.0], "statistic": [1.0, 2.0]})
    assert np.isnan(elasticity(df, target="statistic"))


def test_robustness_check() -> None:
    out = robustness_check(
        BASELINE_KWARGS, axes={"r_g": [0.5, 1.0, 2.0]},
        metric="delta_pp", n_runs=64, seed=1,
    )
    assert isinstance(out, pd.DataFrame)
    assert {"axis", "sign_preserved", "ac10_passed"}.issubset(out.columns)
