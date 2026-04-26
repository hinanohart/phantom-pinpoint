"""High-level agent wrappers (Subject / Audience / Trigger).

The :class:`PhantomPinpointModel` already vectorises the entire population for
speed.  These thin classes are provided for *pedagogical* clarity — they make
the conceptual separation between subject, observer, and external trigger
explicit, matching the language used in :doc:`docs/theory.md` and the unit
tests, and are convenient for downstream code that wants to inspect a single
agent's full trajectory.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from phantom_pinpoint.core import (
    bayesian_posterior_mean,
    pp_divergence,
    strategic_claim,
)


@dataclass(slots=True)
class Trigger:
    """An external, exogenous event in :math:`\\Theta`.

    Attributes
    ----------
    location:
        ``(d,)`` coordinate.
    salience:
        Multiplicative scalar passed to the likelihood width — high salience
        → narrow likelihood → strong Bayesian update.
    """

    location: NDArray[np.float64]
    salience: float = 1.0

    def likelihood_var(self, base_tau: float) -> float:
        """Effective likelihood variance ``(tau / salience) ** 2``."""
        if self.salience <= 0:
            raise ValueError("salience must be > 0")
        return float((base_tau / self.salience) ** 2)


@dataclass(slots=True)
class Audience:
    """The (optional) public observing the subject.

    Attributes
    ----------
    size:
        Effective number of observers — multiplicative scalar on ``gamma``.
    vigilance:
        Probability ∈ [0, 1] that an observer pursues a counterfactual probe
        ("but were you *really* about to do it?").  Used by experiments
        modelling Condition C6.
    """

    size: float = 0.0
    vigilance: float = 0.0

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("audience size must be >= 0")
        if not (0.0 <= self.vigilance <= 1.0):
            raise ValueError("vigilance must be in [0, 1]")


@dataclass(slots=True)
class Subject:
    """A single PP subject with persistent traits and a short memory.

    The class is meant for trace-level inspection and replay; aggregate
    experiments should rely on :meth:`phantom_pinpoint.core.PhantomPinpointModel.simulate`.

    Parameters
    ----------
    dim:
        Dimensionality of :math:`\\Theta`.
    alpha, beta, gamma:
        Per-agent strategic gains (typically sampled from a Beta).
    prior_mean:
        Initial belief mean :math:`\\mu`.
    prior_var:
        Belief variance :math:`\\sigma_\\psi^2`.
    mu_g, r_g:
        Goal region centre and hard radius.
    memory_size:
        Number of past ``(trigger, claim, delta_pp)`` triplets retained for
        analysis.  Bounded ``deque`` so memory cost stays O(memory_size).
    """

    dim: int
    alpha: float
    beta: float
    gamma: float
    prior_mean: NDArray[np.float64]
    prior_var: float
    mu_g: NDArray[np.float64]
    r_g: float
    memory_size: int = 32
    history: deque[tuple[NDArray[np.float64], NDArray[np.float64], float]] = field(
        default_factory=lambda: deque(maxlen=32),
    )

    def __post_init__(self) -> None:
        if self.memory_size <= 0:
            raise ValueError("memory_size must be > 0")
        # ``deque`` maxlen is set at construction time; mirror it here so the
        # user-supplied memory_size is honoured even when the default factory
        # already fired.
        if self.history.maxlen != self.memory_size:
            self.history = deque(self.history, maxlen=self.memory_size)

    def claim(
        self,
        trigger: Trigger,
        rng: np.random.Generator,
        tau: float = 0.3,
        sigma_floor: float = 0.02,
        sigma_naive: float = 0.5,
    ) -> tuple[NDArray[np.float64], float]:
        """Generate this subject's post-hoc claim for one trigger.

        Returns
        -------
        tuple
            ``(p_star, delta_pp)`` — the claimed point and its PP signature.
        """
        p_star, sigma, anchor = strategic_claim(
            trigger.location,
            self.prior_mean,
            self.mu_g,
            self.r_g,
            self.alpha,
            self.beta,
            self.gamma,
            rng=rng,
            sigma_floor=sigma_floor,
            sigma_naive=sigma_naive,
        )
        post_mean, post_var = bayesian_posterior_mean(
            self.prior_mean, self.prior_var, trigger.location, trigger.likelihood_var(tau)
        )
        delta = pp_divergence(p_star, sigma, anchor, post_mean, post_var)
        self.history.append((np.asarray(trigger.location, dtype=np.float64), p_star, delta))
        return p_star, delta

    def replay(self) -> Iterable[tuple[NDArray[np.float64], NDArray[np.float64], float]]:
        """Iterate over ``(trigger, claim, delta)`` triples in chronological order."""
        return iter(self.history)
