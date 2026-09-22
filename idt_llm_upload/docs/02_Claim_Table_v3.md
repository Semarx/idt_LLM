# Claim Table — v3

**13 September 2026 · Semarx Research**

*Claim-level status. The mathematics is in `00_Frozen_Decisions.md` (F-numbers below refer
to it). Condition-level detail and results live in `03_working_log.csv` (row ids below refer
to it). This file is the index, not the record.*

*Supersedes `closure_tests_sep/Master_Claim_Table_v2.md`, which carried the pooled-token
estimator.*

---

## Status vocabulary

| status | meaning |
|---|---|
| **DERIVED** | settled by mathematics; proof written or sketched |
| **NUMERICALLY VERIFIED** | confirmed over a tested range; no general proof |
| **HELD** | experimental evidence in hand |
| **REQUIRED** | test designed, not run |
| **OPEN** | neither derived nor designed |

---

## 1. Derived and null-class claims

| ID | statement | status | where | CSV |
|---|---|---|---|---|
| **D1** | Under closure, `\|Ω\| ≤ (1/W)·log₂(K−1) + h(1/W)`, with equal-weight window distributions. No assumption on pair lengths. | **DERIVED** — full proof owed | F4 | A01 |
| **D2** | A single-pair breach cannot be separated from closure by any threshold derived from D1, at any W, K, or pair-length distribution. | **DERIVED** | F4 | A02 |
| **D3** | K is not a design lever; band and excursion both carry log₂K and it cancels. | **NUMERICALLY VERIFIED** over K ∈ [128, 128 000] | F5 | A03 |
| **D4** | When removed content matches retained content distributionally, Ω approaches a finite-sample floor independent of breach size. | **NUMERICALLY VERIFIED** over the tested range; closed form owed | F3, F7 | A04, A05 |
| **D5a** | Any transformation preserving the empirical entropy of the compared records is invisible to Ω, regardless of eventwise mismatch. | **DERIVED — definitional** | — | — |
| **D6** | Eventwise and distributional rungs have O(1/W) nulls under closure. | **OPEN** — asserted | — | A06, A07 |
| **D7** | Band width orders with the ladder; only Ω carries a log₂K factor. | **OPEN** — asserted | — | A08 |

**D6 is load-bearing.** C8 claims the distributional rung covers Ω's largest blind spot, and
that rung has no derived null yet.

**D5a is not empirically testable** — Ω sees the records only through their entropies, so the
statement is definitional. C7 carries the empirical content.

## 2. Core empirical claims

| ID | claim | status | settled by | CSV |
|---|---|---|---|---|
| **C1** | Ω stays inside the band under manipulations that change behaviour without breaching the carried-forward region | **REQUIRED** | T2 | P01–P06 |
| **C2** | Ω exits under truncation of distributionally distinct prior pairs | **REQUIRED** | T3 | B01–B03, B11–B13 |
| **C3** | Ω exits under selective compaction | **REQUIRED** | T3 | B08–B10 |
| **C4** | Excursion depends jointly on breach size **and** distinctness | **REQUIRED** | T3 | B04–B07 |
| **C5** | Detection latency is bounded and reportable | **REQUIRED** | T3 | B14 |
| **C6** | Ω does **not** exit under uniform thinning at high overlap | **REQUIRED** — predicted null | T3b | N01, N02 |
| **C7** | Realistic guardrail transformations stay inside the band | **REQUIRED** — predicted null | T3b | N03, N04 |
| **C8** | The distributional rung detects what Ω misses under high-overlap removal | **REQUIRED** | T4 | L01 |
| **C9** | The eventwise rung detects single-pair loss and entropy-preserving substitution | **REQUIRED** | T4 | N05, L02 |
| **C10** | The nesting holds on real transcripts | **REQUIRED** | T4 | L03 |
| **C11** | The band transfers to a second model pair without refitting | **REQUIRED** | T6 | X01–X03 |
| **C12** | The band transfers to a second task without refitting | **REQUIRED** | T6 | X04–X06 |

**C1 has no held evidence.** This follows from retiring the per-turn quantity (F9) and is
stated rather than concealed. T2 is the first evidence for the paper's specificity claim.

**C6 and C7 are predicted nulls.** A positive in C6 falsifies D4; a positive in C7 falsifies
D5a's applicability to realistic transformations. Predicted nulls that can kill a derivation
are the strongest rows in the table.

## 3. Scope statements — not performance claims

One short subsection of the paper. Evidence about what the reading is *not* for.

| ID | statement | status | evidence |
|---|---|---|---|
| **SC1** | Ω is a seam-integrity reading, not a graded dialogue-quality score. A preregistered test found no evidence it tracks turn-level task advancement. | **HELD** | D = −0.0218, p = 0.291; −0.0078 (p = 0.444) with turn index. Null validated: mean p 0.5003, FPR 0.0567, n = 192 |
| **SC2** | A uniform level shift in message length did not move the tested reading. Turn-to-turn variation may still contribute. | **HELD**, narrow | H is 95–97% log-length; turn-to-turn r = +0.682, p = 0.043 |
| **SC3** | Decoder settings and instructed response length did not move the tested reading. | **HELD**, on the retired quantity | 9 decoder settings, Kruskal p = 0.359; instruction levels, length ×6 at p < 0.0001, reading p = 0.346 |

SC3 is explicitly marked as measured on the retired per-turn quantity (F9).

## 4. Observation points

| # | observation point | in scope | role |
|---|---|---|---|
| OP1 | conversation history → carried-forward region | yes | primary |
| OP3 | context-window overflow dropping prior pairs | yes | primary |
| OP4 | tool / agent handoff | conditional | requires tool output in the pair record |
| OP5 | guardrail rewriting of carried-forward content | yes | null-class test |
| OP6 | symbol substitution within carried-forward content | yes | null-class test |
| ~~OP2~~ | ~~retrieval → model context~~ | **no** | outside the compared region (F2) |
| — | user prompt → assembled context | **no** | additions by design (F9) |

## 5. Test sequence

| test | purpose | new data | settles | CSV |
|---|---|---|---|---|
| **T0** | implementation verification against synthetic closure. **Not evidence for C1.** | no | — | I01, I02 |
| **T0b** | analytic: D1 proof, D6 constants, D4 closed form, signal ratio at W = 25 | no | D1, D4, D6, D7 | A01–A11 |
| **T1** | instrumentation and the freeze: K, W, n_min, region delimitation, per-pair logging, exclusions | — | prerequisite | S01–S06 |
| **T2** | specificity — behaviour changes without seam breach | yes | C1 | P01–P06 |
| **T3** | sensitivity — breaches crossed by magnitude × distinctness | yes | C2–C5 | B01–B14 |
| **T3b** | predicted nulls | yes | C6, C7 | N01–N05 |
| **T4** | ladder — all three rungs on every T3/T3b condition | derived | C8–C10 | L01–L03 |
| **T6** | external validity — second model pair, second task | yes | C11, C12 | X01–X06 |

## 6. Blocking questions

| # | question | blocks | status |
|---|---|---|---|
| **B2** | Can the carried-forward region be delimited at assembly and read at delivery in the current harness? | T1 | **open** — decides whether T1 is instrumentation or rebuild |
| **B3** | Does the distributional rung have a derived null (D6)? | C8, C10 | **open** — analytic, can run alongside T1 |
| **B4** | Is D1's full proof written? | the paper | **open** — analytic |
| **B5** | Does the signal-to-band ratio hold at W = 25 as it does at W = 100? | freezing W | **open** — analytic, CSV A11 |

**B1 is closed.** W = 25, by F6 — resolved in an unexpected direction: small windows are
better, not worse.

## 7. Where the paper stands

| | count |
|---|---|
| derived, proof owed | 2 (D1, D2) |
| numerically verified over tested range | 2 (D3, D4) |
| derived definitionally | 1 (D5a) |
| open, load-bearing | 2 (D6, D7) |
| held empirically — scope only | 3 (SC1–SC3) |
| **required, no data** | **12 (C1–C12)** |

CSV: 53 conditions, **5 done, 48 open**. Every performance claim is unrun. The derived layer
is ahead of the empirical layer, which is unusual and should be stated plainly rather than
disguised by carrying forward evidence about a retired quantity.
