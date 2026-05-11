# Anecdote inventory

These are the canonical examples that motivate the Phantom Pinpoint
formalism.  None of them constitute empirical evidence — they are *target
phenomena* that the model is intended to capture qualitatively.  Mapping
each anecdote to model parameters is provided to make the operational
content of the model transparent.

## 1. 「今宿題やろうと思ってたのにママ」("I was just about to do my homework, mom!")

> **Parent:** "Did you do your homework?"
> **Child:** "I was *just* about to do my homework, mom!"

* **Subject \(S\)** — child.
* **\(G\)** — diffuse "I should do my homework today" intent (large \(r_G\)).
* **Trigger \(E\)** — mother walks in.
* **Audience \(\gamma\)** — high (mother is the observer + judge).
* **Strategic claim \(p^*\)** — "right now, this minute, before you
  reminded me".  The agent reports a *very specific* point in \(G\).
* **Bayesian counterfactual** — admitting "I had no specific plan" (which
  would be the rational posterior given a flat prior over the day).

This is the canonical PP signature: large \(r_G\), large \(\gamma\),
audience-driven sharp claim.

## 2. Politician's "we anticipated this all along"

* **\(G\)** — vague "we are competent stewards" mental policy.
* **\(E\)** — unforeseen economic event.
* **\(\gamma\)** — extreme (press corps + voters).
* **\(p^*\)** — claim a specific historical statement that "predicted" \(E\).

Empirically observable proxy: speech-act timestamps.  When the claimed
prior statement post-dates the event, the PP signature is direct evidence
of confabulation.

## 3. Trader's post-hoc "this was within my model"

* **\(G\)** — "my trades respect risk limits".
* **\(E\)** — adverse market move.
* **Defensiveness \(\beta\)** — high (avoid blame).
* **\(p^*\)** — "I positioned for exactly this scenario".

The PP framework predicts that the *narrower* the post-hoc claim, the
*greater* the gap from the agent's actual prior risk-budget posterior.

## 4. TV "あざと可愛い" — calculated cuteness

* **\(G\)** — "I am thoughtfully feminine" persona.
* **\(E\)** — being filmed performing a small gesture.
* **\(\gamma\)** — broadcast audience (very large).
* **\(p^*\)** — explicit verbalisation: "I always do my hair like this".

The performer is rewarded for sharp claims, so we predict
\(\sigma_{conf}\) to be near \(\sigma_{floor}\).

## 5. Sibling fight — "I was about to share, but…"

* **\(G\)** — fairness norm.
* **\(E\)** — younger sibling demanding the toy.
* **\(\beta\)** — high (the older child wants to keep both the toy and the
  moral high ground).
* **\(p^*\)** — "I was *just* about to give it to you anyway".

This is structurally identical to the homework excuse but with two
audiences (sibling + parent) competing for credibility.  The model
predicts that, with two observers of differing prior credibility, the
agent's claim drifts toward the more sceptical observer's posterior.

## 6. SNS "匂わせ" (suggestive insinuation)

* **\(G\)** — vague "I have an interesting life".
* **\(E\)** — opportunity to post.
* **\(\gamma\)** — followers count (continuous variable).

PP predicts that the post-hoc specificity of an insinuation post is a
monotone increasing function of follower count — directly testable via a
scraping study.

## 7. Sports commentary — "exactly the play we drew up"

A coach who claims, after a successful improvisation, that the result was
the strategic plan.  Identical structural form to (2) but in a domain with
publicly recorded "strategic plans" (playbooks).  Falsification of the
claim becomes empirical: compare the post-hoc statement to the recorded
playbook.

## Discriminative predictions across the inventory

| Anecdote | Predicted \(\sigma_{conf}\)        | Predicted \(\Delta_{PP}\) | Empirical proxy                      |
|----------|------------------------------------|---------------------------|--------------------------------------|
| 1        | small (audience large, \(\alpha\) large) | positive                  | child self-report timing             |
| 2        | very small                         | positive                  | timestamp of "predictive" statement  |
| 3        | small                              | positive                  | trader's risk system logs            |
| 4        | very small                         | positive                  | broadcast verbal explicitness        |
| 5        | medium                             | positive (audience-shifted) | mediator interview                  |
| 6        | small (scaling with followers)     | positive                  | post specificity vs follower count   |
| 7        | small                              | positive                  | playbook text vs post-game claim     |

These mappings are **suggestive only** and have not been empirically
validated in this v0.1.0 release.  They are intended to motivate future
behavioural-economics replication work; we do **not** make any claim that
real humans actually conform to the operational definition without a
properly conducted, IRB-approved experiment.
