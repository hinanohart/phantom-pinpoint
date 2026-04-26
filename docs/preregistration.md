# Pre-registration plan

This file is intended to be archived to **OSF Registries** or
**aspredicted.org** before any experiment is run.  Once the OSF DOI is
assigned, replace the placeholder below.

* **OSF DOI:** `TBA — placeholder`
* **Author:** hinanohart
* **Software:** `phantom-pinpoint v0.1.0`
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

## Hypotheses (registered before data collection)

| ID | Pre-registered statement                                              | Test                                  | Decision rule (one-tailed) |
|----|------------------------------------------------------------------------|---------------------------------------|----------------------------|
| H1 | Mean \(\Delta_{PP}\) > 0 in baseline                                   | Bootstrap CI                          | CI low > 0                 |
| H2a| Mean \(\sigma_{conf}\) decreases with \(\gamma\)                        | Spearman                              | \(\rho < -0.5,\; p<0.01\)  |
| H2b| Mean \(\Delta_{PP}\) increases with \(\gamma\)                          | Spearman                              | \(\rho > +0.5,\; p<0.01\)  |
| H3 | post-hoc-fit rate increases with \(r_G\)                               | Diff in means + bootstrap CI          | \(\Delta_{\text{fit}} \ge 0.10\) |
| H4 | A1 (\(\alpha\to0\)) and A3 (\(\gamma=0\)) reduce mean \(\Delta_{PP}\)  | Permutation                           | \(p<0.05\), BH-FDR         |
| H5 | A4 (shuffle G) and A5 (shuffle E) drop post-hoc-fit by \(\ge 0.10\)     | Diff in means + bootstrap CI          | non-overlapping CIs        |
| H6 | Larger \(g_\sigma\) (more "near-miss" triggers) reduces mean \(\Delta_{PP}\) | Spearman                       | \(\rho < -0.5,\; p<0.05\)  |

All p-values are corrected with Benjamini–Hochberg \(q=0.05\) within the
H1–H6 family.

## Acceptance criteria

| ID  | Criterion                                                                                | Status (post-hoc, see [`results/`](../results/)) |
|-----|------------------------------------------------------------------------------------------|---------------------------------------------------|
| AC1 | H1 holds: baseline \(\Delta_{PP}\) bootstrap CI low > 0                                  | PASS                                              |
| AC2 | A6 Bayesian neg-control \(\Delta_{PP}\) bootstrap CI high < 0                            | PASS                                              |
| AC3 | At least 4 of H1–H6 pre-registered direction satisfied (BH-FDR \(q<0.05\))               | **PARTIAL** (4/6 — H2b and H6 fail; details below)|
| AC4 | A4/A5 shuffle ablations reduce post-hoc-fit by \(\ge 50\%\) absolute                     | PASS (A4 −42 %, A5 −85 %)                         |
| AC5 | \(\gamma\uparrow \Rightarrow \sigma_{conf}\downarrow\) (Spearman \(\rho<-0.5,\;p<0.01\)) | PASS                                              |
| AC6 | Code + seed + config reproduces results bit-exactly across machines (Docker)             | PASS (CI runs the full pipeline)                  |
| AC7 | Sensitivity sweep \(\tau,\eta,r_G\) at \(\pm 50\%\) preserves AC1–AC5                    | DEFERRED to v0.2.0                                |

## Honest reporting of pre-registered failures

**H2b (audience drives \(\Delta_{PP}\) up)** failed.  The log-Bayes-factor
turns out to be dominated by the *location mismatch* between the strategic
anchor and the Bayesian posterior mean; \(\gamma\) only modulates the *width*
of the claim kernel, which leaves the location term untouched.  The
phenomenon is captured *correctly* by H2a (\(\sigma_{conf}\) does drop
monotonically with \(\gamma\)).  We retain H2b in the registration record
for future analyses but interpret the failure as a **decomposition**:
audience pressure governs *specificity* (width), whereas \(\beta\) and the
geometry of \(G\) govern *direction* (location).

**H6 (sharper trigger distribution → larger \(\Delta_{PP}\))** failed *with
sign reversal*: \(g_\sigma\) is positively correlated with \(\Delta_{PP}\)
(\(\rho = +1.0\), \(p = 1.4 \times 10^{-24}\)).  The reason, identified
post-hoc and to be re-pre-registered for v0.2.0, is that wider triggers more
often land far from \(G\), at which point the strategic anchor (on the
\(G\) boundary) and the Bayesian posterior mean (near the trigger) diverge
*more*, inflating the log Bayes factor.  This is a **monotonicity reversal**
that is consistent with the PP framework but contradicts the naive
intuition that drove H6.

These failures are reported transparently because R5 of our coding rules
forbids point-estimate Δ-claims and the entire goal of pre-registration is
that null/inverted findings be visible.

## Falsification criteria

The PP framework would be falsified by **any** of the following:

* AC1 fails: baseline \(\Delta_{PP}\) CI contains zero (n=1000+).
* AC2 fails: A6 Bayesian neg-control \(\Delta_{PP}\) CI contains zero.
* DP4 fails: shuffling \(G\) does *not* drop post-hoc-fit.
* The model fits all conditions equally well, which would indicate it is
  unfalsifiable (Popperian criterion).

In v0.1.0 *none* of these falsification criteria triggered.
