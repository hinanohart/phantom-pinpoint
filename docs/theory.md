# Theoretical background

## What is the Phantom Pinpoint Effect?

A subject \(S\) holds an *abstract* goal region \(G \subset \Theta\) — for
example, the diffuse intent "I should do my homework sometime today".  When
an external trigger \(E_t\) (a parent walking in, a politician's question, a
market move) happens to **graze** \(G\), \(S\) retroactively claims to have
been targeting a *specific* point \(p^* \in G\) all along
(parent: "Did you do your homework?" — child: "I was *just* about to start, mom!").

The signature is the conjunction of four sub-phenomena:

1. **Texas-Sharpshooter projection** — the claim point \(p^*\) is the
   closest-point projection of the trigger onto \(G\).
2. **Hindsight bias** — the agent reports higher pre-trigger confidence than
   they actually held.
3. **Confabulation** — the specificity of \(p^*\) is fabricated post-hoc.
4. **Audience-driven cheap-talk** — claim sharpness scales with observer
   pressure \(\gamma\).

PP is the **identifiable conjunction** of (1)–(4) within a single trigger
event, distinguishing it from each component in isolation.

## Generative model

We model an agent with prior \(\psi_t = \mathcal N(\mu, \sigma_\psi^2 I)\),
goal centre \(\mu_G\) and (hard) goal radius \(r_G\).  Given a trigger
\(E_t\), the agent samples its post-hoc claim from one of two competing
models.

### Strategic (PP) model

\[
p^* \mid \text{anchor},\sigma_{conf}
\sim \mathcal N\bigl(\text{anchor},\,\sigma_{conf}^2 I\bigr),
\]
\[
\text{anchor}=(1-\beta)\,\Pi_G(E_t) + \beta\,\mu,
\qquad
\sigma_{conf} = \max\!\bigl(\sigma_{floor},\,
                            \sigma_{naive} / (1+\alpha+\gamma)\bigr).
\]

The closest-point projection \(\Pi_G(E_t)\) clamps the trigger onto the
goal-region boundary; the convex combination with \(\mu\) gives a
defensiveness pull (parameter \(\beta\)); the audience-pressure parameter
\(\gamma\) **shrinks** the claim kernel.  Equation 3 in
[`docs/preregistration.md`](preregistration.md) is the closed-form solution
of the strategic argmax.

### Bayesian (rational) model — *negative control*

\[
p^* \mid E_t \sim \mathcal N(\mu_{\text{post}}, \sigma_{conf}^2 I),
\qquad
\mu_{\text{post}}, \sigma_{\text{post}}^2
= \text{Conjugate update}(\psi_t, E_t, \tau^2).
\]

The Bayesian agent ignores \(G\) entirely and reports the conjugate Gaussian
posterior centred at the data-driven mean.

## The PP signature

We compare the two models with the **log Bayes factor**

\[
\Delta_{PP}(p^*)
= \log \mathcal N(p^*;\text{anchor},\sigma_{conf}^2 I)
- \log \mathcal N(p^*;\mu_{\text{post}},\sigma_{\text{post}}^2 I).
\]

* **Strategic-generated data** ⇒ \(p^*\) is centred on the anchor
  ⇒ \(\Delta_{PP} > 0\) (in expectation).
* **Bayesian-generated data** ⇒ \(p^*\) is centred on \(\mu_{\text{post}}\)
  ⇒ \(\Delta_{PP} < 0\) (in expectation).

The Bayesian model therefore gives a clean **negative control** ablation
(A6): the simulation literally cannot recover a positive PP signal under
A6, by construction.  This is exactly the discriminative criterion required
to escape the "PP is just hindsight bias" critique.

## Identification strategy

In a vignette / behavioural study, neither the analyst nor the subject knows
which generative model produced the claim.  The PP framework lets the
analyst **infer** the generative model from observable features of \(p^*\)
and \(E_t\):

* If \(p^*\) sits at \(\Pi_G(E_t)\) (boundary) but \(E_t\) is far outside
  \(G\), the strategic model dominates.
* If \(p^*\) tracks \(E_t\) regardless of \(G\), the Bayesian model
  dominates.

The shuffle ablations (A4, A5) further break the strategic interpretation
without affecting the Bayesian one, because the Bayesian model never uses
\(G\).

## Limitations

1. **Identifiability degeneracy** — when \(E_t \in G\) and \(\mu \approx \mu_G\),
   anchor ≈ posterior mean and the two models become un-discriminable for
   that draw.  We report the marginal \(\Delta_{PP}\) over a population so
   that this degeneracy is averaged out.
2. **Pure simulation** — no human-subjects data are used in this v0.1.0
   release, sidestepping IRB requirements but also sidestepping any claim
   about whether *real* humans exhibit PP.  See
   [`docs/examples.md`](examples.md) for qualitative anecdotes that motivate
   the formalism.
3. **2-D geometry** — the canonical setting is \(d=2\) (intent-strength × 
   timing).  Higher dimensions are supported but not extensively tested in
   v0.1.0.
4. **Static \(G\)** — we do not model an agent that *redraws* \(G\) after
   the trigger ("Texas Sharpshooter on \(G\) itself"); see open issue #1.
