# D1 — the closure bound, with proof

**13 September 2026 · Semarx Research**

*Closes CSV row A01 and blocker B4, the last one. Every step verified numerically in
`src/verify_d1_proof.py`.*

---

## Statement

Let the declared vocabulary have size `K ≥ 2`. For each conversational pair *i*, let `q_i` be
its token distribution — normalised counts over that vocabulary. For a window of `W ≥ 2`
pairs define

```
q̄_Z(t) = (1/W) Σ_{i=t−W+1}^{t}   q_i          emitted window distribution
q̄_S(t) = (1/W) Σ_{i=t−W}^{t−1}   q_i          received window distribution
Ω(t)   = H(q̄_Z(t)) − H(q̄_S(t))
```

> **Proposition 1 (D1).** If the carried-forward region is closed — the received window
> contains exactly the `W` pairs `q_{t−W} … q_{t−1}` as emitted — then
>
> ```
> |Ω(t)| ≤ (1/W)·log₂(K−1) + h(1/W)
> ```
>
> where `h(x) = −x log₂ x − (1−x) log₂(1−x)`.

No assumption is made about pair lengths, about the pairs being drawn from a common
distribution, or about their supports overlapping.

---

## Proof

**Step 1 — the difference is a single pair swap.**

The two windows share the `W−1` pair distributions `q_{t−W+1} … q_{t−1}`, each carrying
coefficient `1/W` in both mixtures. Those terms cancel exactly:

```
q̄_Z(t) − q̄_S(t) = (1/W)( q_t − q_{t−W} )
```

**Step 2 — total variation is at most 1/W.**

```
TV(q̄_Z, q̄_S) = ½ Σ_k |q̄_Z(k) − q̄_S(k)|
              = (1/W) · ½ Σ_k |q_t(k) − q_{t−W}(k)|
              = (1/W) · TV(q_t, q_{t−W})
              ≤ 1/W
```

since total variation between probability distributions is at most 1. **This step is tight:**
when `q_t` and `q_{t−W}` have disjoint support — routine for sparse token distributions —
`TV(q_t, q_{t−W}) = 1` and the inequality holds with equality.

**Step 3 — the continuity estimate.**

Apply the **Fannes–Audenaert inequality** (Audenaert 2007), the sharp form of Fannes'
continuity bound. For any two distributions `P, Q` on `K` symbols with `T = TV(P,Q)`:

```
|H(P) − H(Q)| ≤ T·log₂(K−1) + h(T)
```

This is the best bound obtainable from `T` alone, and it is attained.

**Step 4 — monotonicity lets us substitute the worst case.**

Let `f(δ) = δ·log₂(K−1) + h(δ)`. Then

```
f′(δ) = log₂(K−1) + log₂((1−δ)/δ)
f′(δ) ≥ 0  ⟺  (K−1)(1−δ) ≥ δ  ⟺  δ ≤ 1 − 1/K
```

So `f` increases on `[0, 1 − 1/K]` and decreases after. Since `W ≥ 2` and `K ≥ 2` give
`1/W ≤ ½ ≤ 1 − 1/K`, the interval of interest lies entirely in the increasing region, and
`T ≤ 1/W` implies `f(T) ≤ f(1/W)`.

Combining Steps 2–4:

```
|Ω| = |H(q̄_Z) − H(q̄_S)| ≤ f(T) ≤ f(1/W) = (1/W)·log₂(K−1) + h(1/W)     ∎
```

---

## Conditions, stated

1. `K ≥ 2`, `W ≥ 2`.
2. **Both windows contain exactly `W` pairs.** This is the closure premise. A breach that
   removes pairs leaves the received window short, the cancellation in Step 1 does not hold,
   and the bound does not apply — which is the point: the breach violates the antecedent.
3. Pairs enter the estimator only if eligible under `n_min` (F7). Eligibility is a property
   of the estimator, not of the bound.

---

## Verification

`src/verify_d1_proof.py`:

**Step 4, monotonicity.** Numerical argmax of `f` over 400,000 points:

| K | argmax | predicted `1 − 1/K` |
|---:|---:|---:|
| 256 | 0.996094 | 0.996094 |
| 32,000 | 0.999969 | 0.999969 |

**Step 3, sharpness.** Extremal pair `P = (1,0,…,0)`, `Q = (1−δ, δ/(K−1), …, δ/(K−1))`:

| K | W | δ | \|ΔH\| attained | bound | gap |
|---:|---:|---:|---:|---:|---:|
| 256 | 25 | 0.0400 | 0.562066 | 0.562066 | 0 |
| 32,000 | 25 | 0.0400 | 0.840922 | 0.840922 | 3.3e−16 |
| 32,000 | 50 | 0.0200 | 0.440755 | 0.440755 | 1.1e−16 |
| 32,000 | 100 | 0.0100 | 0.230451 | 0.230451 | 5.6e−17 |

**Step 2, tightness.** Maximum TV over 300 random windows of arbitrary pair lengths and
supports: exactly `1/W` at W = 10, 25 and 50.

---

## What sharpness means for the paper

The band is **the best bound obtainable from total variation alone**, and both steps that
produce it are individually tight. Two consequences.

**1. Ω's insensitivity is a property of the statistic, not of the proof.** On real
transcripts a 40% truncation produces |Ω| at 0.15× the band (F13). That gap cannot be closed
by a cleverer derivation — nobody can tighten this band and rescue Ω's sensitivity.

**2. A tighter band would cost the calibration-free property.** The bound could be improved by
assuming something further about the content — bounded support per pair, a common source
distribution, a bound on pair-to-pair divergence. Every such assumption is a statement about
the deployment that would have to be justified or measured there. **Under Path B (F11), that
is precisely what is not available.** The band is as tight as a calibration-free reading can
be.

So the trade recorded in F11 is not an artifact of how the work was done. It is structural:

| | derived, universal null | sensitivity to truncation / compaction |
|---|---|---|
| **Ω** | yes, and provably as tight as possible without deployment-specific assumptions | no — 0.15× band |
| **JS** | no — 1.67× spread, not predictable from the summary (F12) | yes — 4.5× its null |

---

## Reference

K. M. R. Audenaert, *A sharp continuity estimate for the von Neumann entropy*,
J. Phys. A 40 (2007) 8127. The classical form is M. Fannes, Commun. Math. Phys. 31 (1973) 291.
Cite the Audenaert form; the Fannes form is looser and would give a wider band.
