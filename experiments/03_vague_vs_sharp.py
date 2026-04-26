"""Experiment 03 — vague vs sharp goal region :math:`G`.

Pre-registered hypothesis H3
----------------------------
Larger goal-region radius :math:`r_G` (more *vague* target) increases the
post-hoc-fit rate, because the closest-point projection always lies within
the larger ball.  Specifically: post_hoc_fit(r_G = 2.0) > post_hoc_fit(r_G =
0.25) by at least 0.10 absolute, with a non-overlapping bootstrap CI.
"""

from __future__ import annotations

import sys

import pandas as pd

from phantom_pinpoint.ablations import BASELINE_KWARGS
from phantom_pinpoint.simulation import run_condition
from phantom_pinpoint.statistics import bootstrap_ci

from experiments._common import (
    ensure_dirs,
    write_manifest,
    write_parquet,
)

R_LEVELS = (0.25, 0.5, 1.0, 2.0)


def main(n_runs: int = 600, seed: int = 42) -> int:
    ensure_dirs()
    frames = []
    summary = []
    for i, r in enumerate(R_LEVELS):
        kwargs = {**BASELINE_KWARGS, "r_g": r}
        df = run_condition(f"r_g={r}", n_runs=n_runs, seed=seed + i, **kwargs)
        frames.append(df)
        ci_fit = bootstrap_ci(df["post_hoc_fit"].to_numpy().astype(float), seed=seed + i)
        ci_delta = bootstrap_ci(df["delta_pp"].to_numpy(), seed=seed + i + 100)
        summary.append(
            {
                "r_g": r,
                "fit_mean": ci_fit.statistic,
                "fit_ci_lo": ci_fit.ci_lo,
                "fit_ci_hi": ci_fit.ci_hi,
                "delta_mean": ci_delta.statistic,
                "delta_ci_lo": ci_delta.ci_lo,
                "delta_ci_hi": ci_delta.ci_hi,
            }
        )
    df_all = pd.concat(frames, ignore_index=True)
    write_parquet(df_all, "03_vague_vs_sharp")

    fit_lo = next(s for s in summary if s["r_g"] == min(R_LEVELS))["fit_mean"]
    fit_hi = next(s for s in summary if s["r_g"] == max(R_LEVELS))["fit_mean"]
    write_manifest(
        "03_vague_vs_sharp",
        {
            "n_runs": n_runs,
            "seed": seed,
            "summary": summary,
            "fit_diff": float(fit_hi - fit_lo),
            "H3_passed": bool(fit_hi - fit_lo >= 0.10),
        },
    )
    print(f"post-hoc fit r=0.25: {fit_lo:.3f}  r=2.0: {fit_hi:.3f}  diff={fit_hi-fit_lo:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
