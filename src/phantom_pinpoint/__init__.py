"""Phantom Pinpoint: post-hoc specificity confabulation simulator.

The Phantom Pinpoint (PP) Effect describes the cognitive phenomenon whereby a
subject who holds an *abstract* goal region :math:`G \\subset \\Theta` claims —
the moment an external trigger :math:`E` happens to graze :math:`G` — that they
were targeting a *specific* point :math:`p^* \\in G` all along.  Examples
include the canonical Japanese excuse 「今宿題やろうと思ってたのにママ」
(parent: "Did you do your homework?" — child: "I was just about to do my
homework, mom!") and politicians' retroactive "this was anticipated" claims.

This package provides:

* A minimal, fully-typed numpy implementation of the PP generator
  (:mod:`phantom_pinpoint.core`).
* Agent-based simulation primitives (:mod:`phantom_pinpoint.agents`,
  :mod:`phantom_pinpoint.simulation`).
* Bootstrap confidence intervals and permutation tests required by the
  pre-registered statistical analysis (:mod:`phantom_pinpoint.statistics`).
* Ablation grids (:mod:`phantom_pinpoint.ablations`) and reproducible figures
  (:mod:`phantom_pinpoint.visualization`).
* A typer-based CLI (``phantom-pinpoint``).

References
----------
* Festinger, L. (1957). *A Theory of Cognitive Dissonance*.
* Bem, D. J. (1972). Self-perception theory. *Adv. Exp. Soc. Psychol.*
* Fischhoff, B. (1975). Hindsight ≠ foresight. *J. Exp. Psychol. Hum. Perc.*
* Nisbett, R. E. & Wilson, T. D. (1977). Telling more than we can know.
* Crawford, V. P. & Sobel, J. (1982). Strategic information transmission.
* Spence, M. (1973). Job market signaling.
* Trivers, R. (2011). *The Folly of Fools*.
* Mercier, H. & Sperber, D. (2011). Why do humans reason?  *BBS*.
"""

from __future__ import annotations

from phantom_pinpoint.core import (
    PhantomPinpointModel,
    SimulationResult,
    bayesian_posterior_mean,
    pp_divergence,
    project_onto_region,
    strategic_anchor,
    strategic_claim,
)
from phantom_pinpoint.decomposition import (
    DECOMPOSITION_TOLERANCE,
    DecompositionResult,
    decompose_delta_pp,
    decompose_simulation,
)
from phantom_pinpoint.effect_size import (
    EffectSizeCI,
    bootstrap_effect_size,
    cliffs_delta,
    cohens_d,
    hedges_g,
)
from phantom_pinpoint.identifiability import (
    DegeneracyReport,
    detect_degeneracy,
)
from phantom_pinpoint.identifiability import (
    assess as assess_identifiability,
)
from phantom_pinpoint.sensitivity import (
    SensitivityCell,
    elasticity,
    robustness_check,
    sensitivity_sweep,
)
from phantom_pinpoint.statistics import (
    BootstrapResult,
    bootstrap_ci,
    permutation_test,
)

__version__ = "0.2.0"
__all__ = [
    "DECOMPOSITION_TOLERANCE",
    "BootstrapResult",
    "DecompositionResult",
    "DegeneracyReport",
    "EffectSizeCI",
    "PhantomPinpointModel",
    "SensitivityCell",
    "SimulationResult",
    "__version__",
    "assess_identifiability",
    "bayesian_posterior_mean",
    "bootstrap_ci",
    "bootstrap_effect_size",
    "cliffs_delta",
    "cohens_d",
    "decompose_delta_pp",
    "decompose_simulation",
    "detect_degeneracy",
    "elasticity",
    "hedges_g",
    "permutation_test",
    "pp_divergence",
    "project_onto_region",
    "robustness_check",
    "sensitivity_sweep",
    "strategic_anchor",
    "strategic_claim",
]
