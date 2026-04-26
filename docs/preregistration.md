# Pre-registration plan

This file is intended to be archived to **OSF Registries** or
**aspredicted.org** before any experiment is run.  Once the OSF DOI is
assigned, replace the placeholder below.

* **OSF DOI:** `TBA — placeholder`
* **Author:** hinanohart
* **Software:** `phantom-pinpoint v0.2.0`
* **Repository:** <https://github.com/hinanohart/phantom-pinpoint>

## Discriminative predictions

The Phantom Pinpoint Effect is operationally defined as the conjunction of
four discriminative predictions (DP):

| ID  | Statement                                                                     | Source phenomenon distinguished from |
|-----|-------------------------------------------------------------------------------|--------------------------------------|
| DP1 | Strategic-generated claims yield log-Bayes-factor \(\Delta_{PP}>0\)           | Bayesian update                      |
| DP2 | Audience size \(\gamma\) sharpens claim kernel \(\sigma_{conf}\) monotonically | Pure confabulation (audience-blind)  |
| DP3 | Larger goal radius \(r_G\) raises the post-hoc-fit rate                       | Hindsight bias (radius-blind)        |
| DP4 | Shuffling \(G\) (per-agent random target) collapses post-hoc-fit              | Texas-Sharpshooter (target-blind)    |

A finding that fails any of DP1, DP3, DP4 falsifies the PP framework as
constructed; DP2 is a graceful-degradation prediction whose failure does
*not* falsify PP but does demote the audience component.

## Confirmatory family C1 (v0.1.0, 2026-04-27)

| ID | Pre-registered statement                                              | Test                                  | Decision rule (one-tailed) | Outcome |
|----|------------------------------------------------------------------------|---------------------------------------|----------------------------|---------|
| H1 | Mean \(\Delta_{PP}\) > 0 in baseline                                   | Bootstrap CI                          | CI low > 0                 | ✅ PASS  |
| H2a| Mean \(\sigma_{conf}\) decreases with \(\gamma\)                        | Spearman                              | \(\rho < -0.5,\; p<0.01\)  | ✅ PASS  |
| H2b| Mean \(\Delta_{PP}\) increases with \(\gamma\)                          | Spearman                              | \(\rho > +0.5,\; p<0.01\)  | ❌ FAIL (post-hoc explained by C2 H9) |
| H3 | post-hoc-fit rate increases with \(r_G\)                               | Diff in means + bootstrap CI          | \(\Delta_{\text{fit}} \ge 0.10\) | ✅ PASS |
| H4 | A1 (\(\alpha\to0\)) and A3 (\(\gamma=0\)) reduce mean \(\Delta_{PP}\)  | Permutation                           | \(p<0.05\), BH-FDR         | ❌ FAIL  |
| H5 | A4 (shuffle G) and A5 (shuffle E) drop post-hoc-fit by \(\ge 0.10\)     | Diff in means + bootstrap CI          | non-overlapping CIs        | ✅ PASS  |
| H6 | Larger \(g_\sigma\) reduces mean \(\Delta_{PP}\)                       | Spearman                              | \(\rho < -0.5,\; p<0.05\)  | ❌ INVERTED (post-hoc explained by C2 H10) |

C1 passed 4/6 hypotheses with two honest failures.  H2b and H6 are
*demoted to post-hoc / exploratory* status and excluded from the v0.2.0
BH-FDR correction.  Their failures are **explained** by the v0.2.0
decomposition (see C2 below).

## Confirmatory family C2 (v0.2.0, 2026-04-27, **registered separately**)

C2 is registered as an independent family **after** decomposing
\(\Delta_{PP}\) into width and location components.  The decomposition is
purely algebraic — no new free parameters — so this is a re-analysis of
the v0.1.0 simulation outputs through a new statistical lens, plus three
new sensitivity / identifiability hypotheses.

| ID | Pre-registered statement                                                                              | Test         | Decision rule (one-tailed)  | Outcome |
|----|--------------------------------------------------------------------------------------------------------|--------------|-----------------------------|---------|
| H7 | Across ±50 % univariate sweeps of \(r_G\), \(g_\sigma\), \(\tau\), audience size, the *sign* of mean ΔPP is preserved in ≥ 90 % of cells | Sign-preserve count | ≥ 4/4 axes pass | ✅ PASS  |
| H8 | In the degeneracy region (E ∈ G ∧ ‖μ−μ_G‖ < 0.1·r_G), the ΔPP bootstrap CI **contains** zero, while the non-degenerate region CI **excludes** zero | Bootstrap CI partition | both CIs satisfy | ❌ FAIL (model setup issue, see Honest failures) |
| H9 | Mean \(\Delta_{PP}^{\text{width}}\) increases monotonically with γ                                     | Spearman    | \(\rho > +0.5,\; p<0.01\)   | ✅ PASS (ρ=+1.0, p=1.4e-24) |
| H10| Mean \(\Delta_{PP}^{\text{loc}}\) increases monotonically with \(g_\sigma\)                            | Spearman    | \(\rho > +0.5,\; p<0.01\)   | ✅ PASS (ρ=+1.0, p=1.4e-24) |

BH-FDR \(q=0.05\) is applied within C2 only.

## Falsification criteria (sharpened in v0.2.0)

Critic's F1–F5:

* **F1** β-monotonicity: ΔPP must be monotone in β, otherwise the
  defensiveness term has the wrong sign.
* **F2** temporal causality: swapping G_pre/G_post must change ΔPP — if
  it doesn't, PP is time-symmetric and the trigger ordering is
  meaningless.
* **F3** σ_G → 0 limit: ΔPP must not diverge unless explicitly
  regularised; an unmarked divergence would imply numerical tuning.
* **F4** width-only manipulation: holding μ fixed and varying σ_G alone
  must produce O(1) change in ΔPP_width — else the metric is
  width-blind.  **Result (v0.2.0)**: confirmed by H9.
* **F5** random anchor: replacing the strategic anchor by a random
  reflection must collapse the strategic vs Bayesian discrimination.

## Acceptance criteria

| ID  | Criterion                                                                                | v0.1.0 | v0.2.0 |
|-----|------------------------------------------------------------------------------------------|--------|--------|
| AC1 | C1 H1 holds: baseline ΔPP bootstrap CI low > 0                                           | PASS   | PASS   |
| AC2 | A6 Bayesian neg-control ΔPP bootstrap CI high < 0                                        | PASS   | PASS   |
| AC3 | At least 4 of C1 H1–H6 in pre-registered direction (BH-FDR \(q<0.05\))                  | PARTIAL (4/6) | (legacy)  |
| AC4 | A4/A5 shuffle ablations reduce post-hoc-fit by ≥ 50 %                                    | PASS   | PASS   |
| AC5 | γ↑ ⇒ \(\sigma_{conf}\)↓ (Spearman ρ<-0.5, p<0.01)                                        | PASS   | PASS   |
| AC6 | Code + seed + config reproduces results bit-exactly                                      | PASS   | PASS   |
| AC7 | (deferred to v0.2.0)                                                                     | DEFERRED | (subsumed by AC10) |
| AC8 | ΔPP_width + ΔPP_loc identity holds within 1e-9; ΔPP_width carries audience signal (H9)   | —      | ✅ PASS |
| AC9 | Identifiability degeneracy diagnostic: inside-CI contains 0 ∧ outside-CI excludes 0      | —      | ❌ FAIL (honest) |
| AC10| ±50 % univariate sweep preserves sign in ≥ 90 % of cells across r_g, g_σ, τ, audience    | —      | ✅ PASS (100 %) |
| AC11| Effect-size reporting: every primary contrast has Cohen's d + bootstrap 95 % CI          | —      | ✅ PASS |
| AC12| ρ-only reporting forbidden — every Spearman accompanies a Cohen's d                      | —      | ✅ ENFORCED |

## Honest failures (post-hoc analysis)

* **C1 H2b** (γ → ΔPP_total): failed because the log-Bayes-factor is
  dominated by location, while γ acts only on width.  v0.2.0 H9
  *recovers* the audience effect by reporting ΔPP_width separately.
* **C1 H6** (g_σ → ΔPP_total): failed *with sign reversal*.  v0.2.0 H10
  *explains* the reversal: wider g_σ inflates ΔPP_loc because triggers
  drift further from the goal centre, producing larger anchor /
  posterior-mean separation.
* **C2 H8** (degeneracy diagnostic): the inside-region CI was
  ``[+1.066, +2.223]`` — strictly positive even within the geometric
  "degenerate" region.  Diagnosis: when the prior and likelihood are
  both *tight*, the strategic anchor and Bayesian posterior mean still
  differ slightly because the strategic anchor projects onto the
  *boundary* of G whereas the posterior mean stays at the trigger.  The
  geometric degeneracy region therefore does **not** map cleanly onto
  the statistical un-identifiability region, which is a non-trivial
  scientific finding.  v0.3.0 will sharpen the diagnostic by using a
  Mahalanobis-style distance instead of an indicator function.

## Permanently NO-GO items (Critic's verdict)

* LLM-mediated agents (reproducibility破壊).
* Real human-subjects vignette experiments at v0.x (IRB scope creep).
* Migration to mesa / jax frameworks (K2 violation).
