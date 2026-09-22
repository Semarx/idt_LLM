# Frozen Decisions — Seam Integrity Paper

**13 September 2026 · Semarx Research**

*Authoritative record. Every decision below is frozen. Where a decision is reversed, it is
recorded here with a date and a reason — never silently changed. Other documents in this
folder reference this one rather than restating it.*

*All numbers are reproduced by `src/verify_estimator.py`, which uses no experimental data.*

---

## F1 · Scope — three questions, nothing else

1. What is the exact LLM return-path seam on which closure is the specification?
2. Can Ω detect informational imbalance at that seam without a fitted healthy corpus or a
   task oracle?
3. What can Ω *not* detect, and how does that null class compare with the distributional and
   eventwise readings of the same seam?

A result that does not serve one of these three moves to supporting material or leaves.

## F2 · The seam

One seam: the loop's return path. Compared **across a turn boundary**, never same-event.

```
pair_t   = (p_t, r_t)                      prompt sent, response returned
context_t = [system] + [carried-forward region] + [p_t] + [anything inserted]
```

The **carried-forward region** is the span specified to hold `pair_1 … pair_{t−1}` unchanged.
It is the only part of the context this work reads.

| element | compared? |
|---|---|
| prior pairs | **yes** |
| system prompt, policies, templates | no — added by design |
| current prompt `p_t` | no — not yet carried forward |
| retrieval inserted at turn *t* | no — belongs to no prior pair |

**Consequence, accepted:** retrieval injection is out of scope. Not missed — outside the
compared region by construction.

## F3 · The estimator — equal pair weighting

**Superseded:** pooled token counts across the window. That estimator does not satisfy
TV ≤ 1/W and the failure reaches D1, not merely D2.

```
q_i(k)  = count of token k in pair i / total tokens in pair i
q̄_Z(t) = (1/W) Σ_{i=t−W+1}^{t}   q_i          emitted window distribution
q̄_S(t) = (1/W) Σ_{i=t−W}^{t−1}   q_i          received window distribution
Ω(t)    = H(q̄_Z(t)) − H(q̄_S(t))
```

Each conversational pair is **one observation**, weighted equally regardless of length. A
500-token turn does not receive ten times the weight of a 50-token turn. The tokenizer fixes
the alphabet inside each pair; pair length does not fix pair weight.

**Terminology:** *pair token distribution*, *window-averaged distribution*, *emitted window
distribution*, *received window distribution*. Not "token bag" — it suggests pooling.

### Why it is required

W = 25, K = 2000, one 3000-token pair among 30-token pairs:

| estimator | TV | 1/W |
|---|---|---|
| pooled tokens | **0.2627** | 0.0400 |
| equal-weight pairs | **0.0267** | 0.0400 |

## F4 · The bound

The windows share W−1 pair distributions, each with coefficient 1/W in both mixtures, so
those terms cancel exactly:

```
q̄_Z − q̄_S  =  (1/W)(q_t − q_{t−W})

TV(q̄_Z, q̄_S)  =  (1/W) · TV(q_t, q_{t−W})  ≤  1/W
```

since total variation between probability distributions is at most 1. **No assumption about
pair lengths anywhere.**

> **D1.** Under closure of the carried-forward region, with window distributions defined as
> equal-weight mixtures of per-pair token distributions,
> ```
> |Ω| ≤ (1/W)·log₂(K−1) + h(1/W)
> ```
> *Full proof owed: the exact continuity estimate invoked and its conditions.*

> **D2.** A single-pair breach displaces the received window distribution by
> `(q̄_others − q_j)/W`, giving TV ≤ 1/W — the same total variation closure itself produces.
> The excursion ceiling is therefore identical to the D1 band, and no threshold derived from
> D1 separates a one-pair breach from closure, **at any W, any K, and any pair-length
> distribution.**

The "pair masses must be comparable" condition carried in earlier versions is **removed**.
It was an artifact of pooling.

## F5 · K is declared, not tuned

Band and excursion both carry log₂K and the factor cancels. W = 100, m = 20, overlap 0:

| K | 128 | 1,280 | 12,800 | 128,000 |
|---|---|---|---|---|
| ratio to band | 2.38 | 2.77 | 2.66 | 2.31 |

Flat across a 1000× sweep. **Coarsening the alphabet is not a remedy for a wide band.** The
tokenizer is a free declaration.

## F6 · W = 50, conversations of 60 turns

**Amended 13 Sep 2026.** The first version of F6 set W = 25 on the argument that an *m*-pair
breach gives a signal-to-band ratio depending on *m* alone, so small W minimised noise-to-band
at no cost in signal. **That argument used the excursion *ceiling*, which scales with the
band. The actual excursion does not.**

Measured: a 40% distinct breach produces |Ω| ≈ 0.88 bits at **every** window length from
W = 10 to W = 100. The excursion is a property of the content difference, not of the bound.
Signal-to-band therefore scales *with* W.

Minimum breach fraction that clears the band, overlap = 0 (values are |Ω|/band; >1 clears):

| W | band | 10% | 20% | 30% | 40% | 50% | 60% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.841 | 0.34 | 0.68 | 0.92 | **1.03** | 1.14 | 1.31 |
| 50 | 0.441 | 0.80 | **1.31** | 1.71 | 1.98 | 2.28 | 2.43 |
| 75 | 0.302 | **1.24** | 1.95 | 2.45 | 2.94 | 3.30 | 3.45 |
| 100 | 0.231 | **1.58** | 2.56 | 3.30 | 3.87 | 4.23 | 4.43 |

Signal-to-noise moves the same way: 5.7 at W=25, 7.6 at W=50, 9.6 at W=100. **Both criteria
favour larger W.**

**W = 50.** Clears a 20% breach, a 40% breach at 2× margin, S/N 7.6. W = 75 and 100 detect
better but cost proportionally more to generate.

**Conversations run 60 turns.** The window needs W pairs before it yields a reading, so the
conversation must exceed W. Sixty turns gives ten per-turn readings per run at W = 50.

**Consequence, accepted:** the eight existing 25-turn conversations cannot be used. Generation
cost rises from ~10 to ~25 minutes per run.

*W is the sliding window inside a conversation. It is not the conversation length. The first
version of F6 conflated the two.*

## F7 · Estimator eligibility — n_min

**Minimum pair length, not smoothing.** Smoothing injects probability mass that was not in
the conversation, makes H and Ω less direct, and adds a parameter to defend. A minimum-length
rule only says: we do not estimate a pair-level distribution from too few observations.

> **Rule.** A pair enters the window estimator only if its token count is at least `n_min`.
> `n_min` is fixed before T1 by synthetic calibration on **estimator reliability alone** —
> two independent realisations of identical pair distributions, true Ω = 0 by construction —
> as the smallest pair length whose false-excursion rate against the theoretical band is
> ≤ 5% at the declared W and K. It is never lowered afterwards. The count of excluded real
> pairs is reported.

Calibration target note: **true closure is the wrong target.** Under true closure the two
windows share W−1 pairs that are the *same estimates*, so their noise cancels and the
false-excursion rate is 0.000 at every pair length. Independent re-realisation isolates
estimator noise with no breach and no detection performance involved.

**Decision criterion, made explicit 13 Sep 2026.** A cell qualifies only if the **Wilson 95%
upper bound** on the false-excursion rate is ≤ tolerance — not merely the point estimate. A
rate estimated at exactly the tolerance carries real probability of exceeding it.

Recheck at 3,000 trials (W=50) and 1,500 (W=75, 100), K = 32,000:

| W | band | n | FE rate | Wilson 95% CI | verdict |
|---:|---:|---:|---:|---|---|
| 50 | 0.441 | 5 | 0.0457 | [0.0388, **0.0537**] | fail — upper bound over tolerance |
| 50 | 0.441 | **10** | **0.0170** | [0.0130, 0.0223] | **PASS → n_min** |
| 75 | 0.302 | 10 | 0.0687 | [0.0569, 0.0826] | fail |
| 75 | 0.302 | **15** | 0.0253 | [0.0185, 0.0346] | **PASS → n_min** |
| 100 | 0.231 | 15 | 0.0620 | [0.0509, 0.0754] | fail |
| 100 | 0.231 | **20** | 0.0300 | [0.0225, 0.0399] | **PASS → n_min** |

**At K = 32,000 and W = 50: n_min = 10.**

The earlier 300-trial sweep put n = 5 at exactly 0.050 at W = 50 and read it as passing. At
3,000 trials it is 0.0457 with an upper bound of 0.0537 — **it fails.** Excluding pairs below
10 tokens removes essentially nothing from real transcripts, where pairs run 80+ tokens.

**Framing:** theory fixes the band; an independently calibrated minimum sample size fixes
when the empirical estimator is reliable enough to apply it. This is finite-sample estimation
variance, not a limitation of D1.

## F8 · Freeze order

```
declare K (tokenizer)  →  declare W  →  calibrate n_min at that (W,K)  →  T1
```

`n_min` depends on both. It cannot be set before K and W.

Current freeze: **K = 32,000 · W = 50 · conversations 60 turns · n_min = 10.**

## F9 · Retired

| item | date | reason |
|---|---|---|
| bipredictability / P | Sep 2026 | pooled bags give a mixture over one token variable, not a joint; MI identities do not apply |
| the same-event transmission check | Sep 2026 | identity null, no theorem; not a closure reading |
| *user prompt → assembled context* observation point | Sep 2026 | additions are by design; closure is not its specification |
| retrieval injection (OP2) as a breach | Sep 2026 | outside the compared region under F2 |
| the per-turn first difference `H(m_t) − H(m_{t−1})` | Sep 2026 | not the seam quantity; cannot inform it |
| the three experiments measuring it | Sep 2026 | retired as evidence; survive as scope-setting only |
| pooled-token window estimator | Sep 2026 | breaks TV ≤ 1/W; see F3 |
| "pair masses must be comparable" condition on D2 | Sep 2026 | artifact of pooling; removed by F3 |
| "tighten the band with a longer window" | Sep 2026 | first reversed by F6, then **re-reversed 13 Sep**: larger W IS better. See F6 amendment |
| "tighten the band with a coarser alphabet" | Sep 2026 | reversed by F5 |
| D6 — O(1/W) null for the distributional rung | 13 Sep 2026 | refuted; see F12 |
| Ω as the paper's primary detector | 13 Sep 2026 | blind to truncation and compaction on real content (F13); the trade in F11 is the result instead |

**Cost of retiring the per-turn quantity, accepted:** C1 — the specificity claim — has no
held evidence. T2 is its first evidence.

## F10 · Standing method rules

1. Every null validated on pure noise before any real result is read.
2. Ω is never shuffled. Nulls permute labels or shuffle H and re-difference.
3. Preregistered items frozen and approved before execution; secondary analyses labelled
   secondary and never substituted for the frozen result.
4. Exclusions declared before running and reported with counts.
5. Numerical conclusions are never carried across a change of estimator without
   recomputation.

## F11 · Path B — the observability premise

**Adopted 13 Sep 2026, on the D6 findings (`04_D6_Findings.md`).**

The observer sees **window-level distributional summaries of each side and cannot align
them** — the application logs a running window histogram, the downstream side exposes
aggregate statistics, or the pipeline reorders. Per-pair correspondence is not available.

**Why the premise is necessary.** With aligned per-pair records, closure means the two
records are the same text: no sampling, no estimation, null exactly zero for every rung, and
any difference is a breach. A hash then does the job more cheaply and Ω has no role. The 1/W
band is the price of not aligning; stated without the premise, the theorem solves a problem
that does not exist.

**What the premise costs.**

1. **The eventwise rung is outside it.** It requires alignment. The ladder is two rungs.
2. **Ω's blind spots lose their cover.** Single-pair loss (D2) and entropy-preserving
   substitution (D5a) become undetectable by any available reading — a characterisation, not
   a gap to be filled.

**The central result under F11.** On real conversational content, the two available readings
trade against each other and neither gives both properties:

| | derived, universal null | sensitivity to truncation / compaction |
|---|---|---|
| **Ω** | **yes** — D1, identical on every deployment | **no** — 0.15× band |
| **JS** | **no** — 1.67× spread, not predictable from the summary | **yes** — 4.5× its null |

## F12 · D6 is refuted

The distributional rung does **not** have a derived O(1/W) null. Measured under closure on
eight real transcripts:

- scales as **W^−1.4**, not 1/W — support sparsity puts JS outside the local-quadratic regime
- **not universal**: 1.67× spread across conversations (Ω's band: identical for all eight)
- **not computable** from the window summary: best predictor R² = 0.375, n = 8, p = 0.107

Closes blocker B3 by refutation. Full record in `04_D6_Findings.md`.

## F13 · D4 widened

Ω's null class is **entropy similarity**, not distributional similarity. Turns with pairwise
JS of 0.39–0.52 — substantially distinct — still produce |Ω| of only 0.13 bits when 40% are
removed. D4's simulation used a permuted vocabulary for removed content, a disjoint token set
that does not occur in real dialogue. Most natural conversation lies inside this null class.
