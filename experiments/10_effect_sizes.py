"""Experiment 10 — effect sizes for every primary contrast.

Computes Cohen's d (with bootstrap 95 % CI) and Cliff's δ for the four
primary contrasts of v0.2.0:

* baseline_strategic vs A6 Bayesian neg-control
* audience γ=0  vs γ=4
* r_g=0.25 vs r_g=2.0
* shuffleG vs shuffleE (cross-ablation)
"""

from __future__ import annotations

import sys

from experiments._common import (
    ensure_dirs,
    write_manifest,
)
from phantom_pinpoint.ablations import ABLATIONS, BASELINE_KWARGS, ablation_grid
from phantom_pinpoint.effect_size import (
    bootstrap_effect_size,
    cliffs_delta,
    cohens_d,
    hedges_g,
)
from phantom_pinpoint.simulation import run_condition


def main(n_runs: int = 1000, seed: int = 42) -> int:
    ensure_dirs()
    grid = ablation_grid(n_runs=n_runs, seed=seed)
    contrasts = []

    base = grid.loc[grid["ablation"] == "baseline", "delta_pp"].to_numpy()
    bayes = grid.loc[grid["ablation"] == "A6_bayesian", "delta_pp"].to_numpy()
    sg = grid.loc[grid["ablation"] == "A4_shuffleG", "delta_pp"].to_numpy()
    se = grid.loc[grid["ablation"] == "A5_shuffleE", "delta_pp"].to_numpy()

    contrasts.append(
        (
            "baseline_vs_bayesian",
            bootstrap_effect_size(
                base, bayes, statistic=cohens_d, n_resamples=3000, seed=seed,
            ),
            float(cliffs_delta(base, bayes)),
            float(hedges_g(base, bayes)),
        )
    )
    contrasts.append(
        (
            "shuffleG_vs_shuffleE",
            bootstrap_effect_size(
                sg, se, statistic=cohens_d, n_resamples=3000, seed=seed + 1,
            ),
            float(cliffs_delta(sg, se)),
            float(hedges_g(sg, se)),
        )
    )

    private = run_condition(
        "private", n_runs=n_runs, seed=seed + 100,
        **{**BASELINE_KWARGS, "audience_size": 0.0},
    )["delta_pp"].to_numpy()
    public = run_condition(
        "public", n_runs=n_runs, seed=seed + 100,
        **{**BASELINE_KWARGS, "audience_size": 4.0},
    )["delta_pp"].to_numpy()
    contrasts.append(
        (
            "private_vs_public_delta_pp",
            bootstrap_effect_size(
                private, public, statistic=cohens_d, n_resamples=3000, seed=seed + 2,
            ),
            float(cliffs_delta(private, public)),
            float(hedges_g(private, public)),
        )
    )

    sharp = run_condition(
        "sharp_g", n_runs=n_runs, seed=seed + 200,
        **{**BASELINE_KWARGS, "r_g": 0.25},
    )["delta_pp"].to_numpy()
    vague = run_condition(
        "vague_g", n_runs=n_runs, seed=seed + 200,
        **{**BASELINE_KWARGS, "r_g": 2.0},
    )["delta_pp"].to_numpy()
    contrasts.append(
        (
            "sharp_vs_vague_g",
            bootstrap_effect_size(
                sharp, vague, statistic=cohens_d, n_resamples=3000, seed=seed + 3,
            ),
            float(cliffs_delta(sharp, vague)),
            float(hedges_g(sharp, vague)),
        )
    )

    summary = []
    for name, ci, delta, g in contrasts:
        summary.append(
            {
                "contrast": name,
                "cohens_d": ci.statistic,
                "cohens_d_ci_lo": ci.ci_lo,
                "cohens_d_ci_hi": ci.ci_hi,
                "cohens_d_significant": ci.is_significant,
                "cliffs_delta": delta,
                "hedges_g": g,
                "n_a": ci.n_a,
                "n_b": ci.n_b,
            }
        )
        print(
            f"{name:>30}  d={ci.statistic:+.2f} [{ci.ci_lo:+.2f}, {ci.ci_hi:+.2f}]"
            f"  δ={delta:+.2f}  g={g:+.2f}  sig={ci.is_significant}"
        )

    write_manifest(
        "10_effect_sizes",
        {
            "n_runs": n_runs,
            "seed": seed,
            "ablations": list(ABLATIONS.keys()),
            "contrasts": summary,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
