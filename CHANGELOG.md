# Changelog

All notable changes to this project are documented in this file.  The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] – 2026-04-27

### Added

* **`decomposition` module** — closed-form decomposition of the log
  Bayes factor into a width term and a location term:
  \(\Delta_{PP} = \Delta_{PP}^{\text{width}} + \Delta_{PP}^{\text{loc}}\).
  Property-based tests (hypothesis) verify the sum-to-whole identity to
  numerical precision.
* **`effect_size` module** — Cohen's *d*, Hedges' *g*, Cliff's δ with a
  generic `bootstrap_effect_size` wrapper.  Critic's AC11/AC12 enforced
  across the v0.2.0 manifests.
* **`sensitivity` module** — `sensitivity_sweep`, `elasticity`,
  `robustness_check` for AC10 (formerly deferred AC7).  ±50 % univariate
  sweep sign-preservation gate.
* **`identifiability` module** — geometric degeneracy diagnostic:
  `detect_degeneracy`, `assess` returning a `DegeneracyReport`.
* **Experiment 06 — sensitivity sweep**: AC10 gate.
* **Experiment 07 — identifiability diagnostic**: AC9 gate.
* **Experiment 09 — location decomposition**: H2b / H6 root-cause via
  the new decomposition, both H9 and H10 PASS at ρ=+1.0, p=1.4e-24.
* **Experiment 10 — effect-size sheet**: every primary contrast now
  reports Cohen's *d* with bootstrap 95 % CI, Cliff's δ and Hedges' *g*.
* **`docs/preregistration.md`** rewritten with:
  * Confirmatory family C1 (v0.1.0) — hold-out from v0.1.0.
  * Confirmatory family C2 (v0.2.0) — H7–H10, registered separately.
  * Falsification criteria F1–F5 (Critic).
  * Honest failures section with H2b / H6 / H8 root-cause discussion.
* **CHANGELOG.md** (this file).

### Changed

* `__version__` bumped to `"0.2.0"` and surfaced in
  `pyproject.toml` and `CITATION.cff`.
* `pp_divergence` is now joined by the equivalent decomposition path —
  both are kept; the decomposed form is recommended for *all* new
  research code because it isolates the audience-driven specificity
  signal that v0.1.0 H2b missed.
* `__init__.py` re-exports the v0.2.0 public surface (additive only).
* CI raised: matrix unchanged but coverage gate stays at 75 % to
  accommodate the larger codebase (test count 53 → 110+, coverage held).

### Fixed

* No critical fixes — v0.1.0 results reproduce bit-for-bit.

### Honest failures (carried over)

* **C1 H2b** (audience → ΔPP_total): was failure, now *explained* by
  decomposition (audience shows up in ΔPP_width, not ΔPP_total).
* **C1 H6** (g_σ → ΔPP_total): was inverted, now *explained* (location
  term inflates with σ_G).
* **C2 H8** (degeneracy → CI contains 0): newly discovered honest
  failure of the geometric diagnostic; the diagnostic does not align
  with statistical un-identifiability when the prior is sufficiently
  tight.  v0.3.0 will sharpen with a Mahalanobis variant.

### Permanently NO-GO

* LLM-mediated agents (reproducibility).
* Real human-subjects vignette study (IRB scope creep).
* Migration to mesa / jax (K2 violation).

## [0.1.0] – 2026-04-27 (initial release)

### Added

* Five governing equations: closest-point projection, Bayesian conjugate
  posterior, strategic argmax, confabulation kernel, log Bayes factor
  ΔPP.
* `core`, `agents`, `simulation`, `statistics`, `ablations`,
  `visualization`, `cli`, `_logging` modules.
* 53 tests, 92 % coverage, 5 reproducible experiments, atomic Parquet
  writes, JSON manifests.
* GitHub Actions CI on Python 3.11 + 3.12.
* MIT licence, CITATION.cff, pre-registration document.

### Acceptance criteria (v0.1.0)

AC1, AC2, AC4, AC5, AC6 PASS.  AC3 partial (4/6).  AC7 deferred.
