"""Experiment 07 — identifiability degeneracy diagnostic (AC9).

Pre-registered hypothesis H8
----------------------------
Conditional on the *degenerate* region — trigger ∈ G ∧ ‖μ−μ_G‖ < ε·r_G —
the bootstrap 95 % CI for ΔPP **contains** zero (the two competing
generative models are unidentifiable, exactly as predicted analytically).
Conditional on the non-degenerate region the CI **excludes** zero.
"""

from __future__ import annotations

import sys

import numpy as np

from experiments._common import (
    ensure_dirs,
    write_manifest,
    write_parquet,
)
from phantom_pinpoint.ablations import BASELINE_KWARGS
from phantom_pinpoint.core import PhantomPinpointModel
from phantom_pinpoint.identifiability import (
    assess,
    detect_degeneracy,
)
from phantom_pinpoint.simulation import run_condition


def main(n_runs: int = 1500, seed: int = 42) -> int:
    """Use a *very* tight prior so a substantial fraction of agents land in
    the degenerate region (otherwise the n_inside is too small to power the
    bootstrap)."""
    ensure_dirs()
    kwargs = {
        **BASELINE_KWARGS,
        "prior_var": 0.05,   # very tight prior → μ near μ_G with high prob
        "g_sigma": 0.5,      # triggers concentrated near μ_G
    }
    df = run_condition("identifiability", n_runs=n_runs, seed=seed, **kwargs)

    # Reproduce the prior_means used by the simulator.
    rng = np.random.default_rng(seed)
    model = PhantomPinpointModel(**kwargs)
    _alphas = rng.beta(*model.alpha_dist, size=n_runs)
    _betas = rng.beta(*model.beta_dist, size=n_runs)
    _gammas = rng.beta(*model.gamma_dist, size=n_runs) * model.audience_size
    prior_means = np.asarray(model.mu_g, dtype=np.float64) + rng.normal(
        0.0, np.sqrt(model.prior_var), size=(n_runs, model.dim),
    )

    triggers = df[["trigger_x", "trigger_y"]].to_numpy(dtype=np.float64)
    is_deg = detect_degeneracy(triggers, prior_means, np.asarray(model.mu_g), model.r_g)
    df = df.assign(is_degenerate=is_deg)
    write_parquet(df, "07_identifiability")

    rep = assess(df["delta_pp"].to_numpy(), is_deg, seed=seed)
    write_manifest(
        "07_identifiability",
        {
            "n_runs": n_runs,
            "seed": seed,
            "kwargs": dict(kwargs.items()),
            "report": {
                "fraction_degenerate": rep.fraction_degenerate,
                "delta_pp_inside_lo": rep.delta_pp_inside_lo,
                "delta_pp_inside_hi": rep.delta_pp_inside_hi,
                "delta_pp_outside_lo": rep.delta_pp_outside_lo,
                "delta_pp_outside_hi": rep.delta_pp_outside_hi,
                "n_inside": rep.n_inside,
                "n_outside": rep.n_outside,
                "ac9_passed": rep.ac9_passed,
            },
        },
    )
    print(
        f"degenerate fraction: {rep.fraction_degenerate:.3f}  "
        f"inside CI [{rep.delta_pp_inside_lo:+.3f}, {rep.delta_pp_inside_hi:+.3f}]  "
        f"outside CI [{rep.delta_pp_outside_lo:+.3f}, {rep.delta_pp_outside_hi:+.3f}]  "
        f"AC9 {'PASS' if rep.ac9_passed else 'FAIL'}"
    )
    return 0 if rep.ac9_passed else 1


if __name__ == "__main__":
    sys.exit(main())
