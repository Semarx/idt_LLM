# D6 — the distributional rung's null, and what it settles

**13 September 2026 · Semarx Research**

*Closes CSV rows A06 and A07 and blocker B3. Measured on the eight 25-turn advancement
transcripts, in which closure holds by construction. No new generation.*

---

## Why D6 was taken

The real-transcript check (§1 below) showed Ω cannot see the breaches the paper is built on.
The proposed rescue was to promote the distributional rung, which moved 8.7× on the same
breach. That required D6: **does JS between the two window distributions have a derived,
universal null under closure?**

It does not. Four findings.

---

## 1. The finding that forced the question

Excursion on real transcripts, 40% of the window breached, mean over 8 conversations:

| breach | mean \|Ω\| | W=25 | W=50 | W=100 | W=200 |
|---|---:|---:|---:|---:|---:|
| **truncate** | 0.127 | 0.15 | 0.29 | 0.55 | 1.06 |
| **compact** | 0.106 | 0.13 | 0.24 | 0.46 | 0.88 |
| rewrite (token-identity change) | 0.787 | 0.94 | 1.79 | 3.41 | 6.54 |

Ratios are |Ω|/band. **Truncation and compaction are invisible to Ω at every achievable
window.** W = 200 requires 250-turn conversations and still gives 1.06 and 0.88.

Pairwise JS between turns in these transcripts is **0.39–0.52** — turns are substantially
distinct. These are not similarity-matched breaches. D4's simulation used a *permuted
vocabulary* for removed content, a disjoint token set that does not occur in real dialogue.

**Ω's null class is therefore larger than D4 stated:** not "content similar to what remains"
but *"content with similar entropy to what remains"*, which describes most natural
conversation.

## 2. JS's null does not scale as O(1/W)

Measured under closure:

| W | 8 | 12 | 16 | 20 | 24 |
|---|---:|---:|---:|---:|---:|
| JS_null | 0.0308 | 0.0191 | 0.0124 | 0.0085 | 0.0067 |
| × W | 0.246 | 0.229 | 0.198 | 0.170 | 0.160 |
| × W² | 1.97 | 2.75 | 3.17 | 3.39 | 3.84 |

Fitted exponent ≈ **W^−1.4**, between 1/W and 1/W². Cause: support sparsity. Each pair
touches ~200 of 32,000 tokens, so consecutive pair distributions are largely disjoint and JS
sits outside the local-quadratic regime in which a clean bound would hold.

**D6 as asserted — O(1/W) for the distributional rung — is false.**

## 3. JS's null is not universal

At W = 24, across the eight conversations:

| | spread | sd/mean |
|---|---|---|
| raw JS_null | **1.67×** | 0.189 |
| normalised by within-window pairwise distinctness | 1.52× | 0.146 |
| **Ω band** | **1.00× — identical for all eight** | **0** |

## 4. And it is not computable from the window summary

Regressing JS_null on quantities available in a window-level summary:

| predictor | Pearson r | p |
|---|---:|---:|
| mean tokens per pair | −0.612 | 0.107 |
| 1 / mean tokens | +0.612 | 0.107 |
| mean types per pair | −0.502 | 0.205 |
| types in window mixture | +0.106 | 0.803 |
| entropy of window mixture | +0.086 | 0.839 |
| concentration Σq² | +0.228 | 0.586 |

Best single predictor gives **R² = 0.375** on n = 8, p = 0.107 — not significant — and reduces
the residual spread only from 1.67× to 1.42×.

**The JS null must be fitted per deployment.** That forfeits the property this work exists to
claim.

## 5. What JS does buy

Breach/null ratio on a 40% truncation, per conversation:

```
3.82  4.33  4.01  3.18  7.53  6.51  5.06  1.64      mean 4.51
```

Against Ω's 0.15. **JS is roughly 30× more sensitive on the breach that matters.**

---

## 6. The structural finding

Tracing where JS's null comes from showed that the 1/W term in **every** rung has one source:
the received window **lags** the emitted window by one pair, and the band tolerates that lag.

In an LLM pipeline the correspondence is known — slot *i* of the carried-forward region is
emitted pair *i*. Align on it and, under closure, the two records are **the same text**:

- no sampling, no estimation, no noise
- the null for every rung is **exactly zero**
- any difference at all is a breach
- and Ω's band, which made truncation invisible, disappears

**The 1/W band is the price of not aligning.** Proposition 1 comes from a setting where two
streams are observed and cannot be matched. Here they can — unless the observer is denied the
per-pair records.

---

## 7. Consequence: the paper takes Path B

**Path A — aligned records.** Null exactly zero, all rungs, truncation and compaction
trivially detected. But a hash does the same thing more cheaply and Ω has no role. A working
monitor and a weak paper. **Rejected.**

**Path B — window-level summaries only. Adopted.** The observer sees distributional summaries
of each side and cannot align them: the application logs a running window histogram, the
downstream side exposes aggregate statistics, or the pipeline reorders. This is the setting in
which the theorem earns its keep, and it must be stated as the premise up front.

### What Path B changes

**The eventwise rung becomes unavailable.** It requires per-pair alignment, which the premise
denies. The ladder collapses from three rungs to two.

**Ω's blind spots lose their cover.** Under Path B, single-pair loss (D2) and
entropy-preserving substitution (D5a) are undetectable by any available reading — not merely
invisible to Ω. That is now a characterisation, not a gap to be filled by a lower rung.

**The two available readings trade against each other:**

| | derived, universal null | sensitivity to truncation / compaction |
|---|---|---|
| **Ω** | **yes** — D1, identical on every deployment | **no** — 0.15× band |
| **JS** | **no** — 1.67× spread, not predictable | **yes** — 4.5× its null |

**Neither gives both.** That is the paper's central result under Path B, and it is a genuine
characterisation of what window-level observability permits.

---

## 8. Status changes

| item | was | now |
|---|---|---|
| **D6** | OPEN — asserted O(1/W) | **REFUTED.** JS null scales W^−1.4, is not universal (1.67×), and is not predictable from summary statistics (R² 0.375, n=8) |
| **B3** | open blocker | **closed by refutation** |
| **C2, C3** | Ω exits under truncation / compaction | **reclassified as predicted nulls for Ω** on homogeneous conversational content, with the measured ratios above |
| **C8, C9, C10** | ladder claims over three rungs | **rewritten for two rungs.** The eventwise rung is outside Path B's premise |
| **D4** | similarity limit | **widened** — the null class is entropy-similarity, not distributional similarity |
| central claim | Ω as calibration-free breach detector | **the trade itself**: under window-level observability, a derived null and sensitivity are not simultaneously available |

## 9. What still needs generation

Much less than before. The characterisation above rests on eight existing transcripts under
one task and one model pair. What T2/T3 now buy is **external validity for the trade**, not
discovery of it:

1. Does the Ω-blind / JS-sensitive split hold on a second task with more heterogeneous turns?
2. Does JS's null spread stay near 1.67× across a second model pair, or widen?
3. Does any breach type move Ω past its band on real content — the rewrite row suggests
   token-identity changes do, at 1.79× for W = 50.

That third row is the one live positive result and it has had no dedicated test.
