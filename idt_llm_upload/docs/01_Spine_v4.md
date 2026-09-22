# Seam Integrity in LLM Pipelines — Paper Spine

**v4 · 13 September 2026 · Semarx Research**

*The argument. Mathematics and frozen parameters are in `00_Frozen_Decisions.md`; claim
status in `02_Claim_Table_v3.md`; condition-level results in `03_working_log.csv`. This
document does not restate them.*

*Supersedes v1–v3 in `closure_tests_sep/`. v3 carried the pooled-token estimator, which
breaks the bound (F3), and guidance to lengthen the window, which is reversed (F6).*

---

## 1. Central claim

> LLM context pipelines can silently alter what is carried forward to the model, and the
> failure is invisible at the behavioural layer until it produces a violation. We define the
> return-path seam at which preservation is the specification — the carried-forward
> conversation record, compared as emitted against as delivered — and test whether
> informational closure gives a calibration-free runtime integrity reading of that seam.
> The contribution is not broad failure detection. It is a null fixed by derivation rather
> than by a fitted healthy corpus, portability across model and task without refitting, and
> an explicit hierarchy of what progressively finer seam readings can and cannot detect.

**Three questions, nothing else** (F1):

1. What is the exact LLM return-path seam on which closure is the specification?
2. Can Ω detect informational imbalance at that seam without a fitted healthy corpus or a
   task oracle?
3. What can Ω *not* detect, and how does that null class compare with the distributional and
   eventwise readings of the same seam?

---

## 2. The gap

### 2.1 The failure mode is documented and consequential

**[Governance Decay](https://arxiv.org/abs/2606.22528)** (preprint v2, June 2026), 1,323
episodes, deterministic violation detection via tool-call parsing:

| condition | violation rate |
|---|---|
| full context (control) | **0%** across all models |
| passive compaction | **30%** pooled, 0–59% by model |
| policy survives the summary | 0% |
| policy dropped by the summary | **38%** |
| optimised summariser injection | Claude 0% → **65%** |

Soft organisational policies decay **8.3×** more than hard safety norms.

The mechanism is content present at turn *t* failing to arrive at turn *t+1* — a return-path
event, with no signal at the behavioural layer until a violation occurs.
**[Lost in Compaction](https://arxiv.org/html/2608.11242)** independently evaluates
side-constraint loss under compaction.

### 2.2 No detector exists for it

Governance Decay: *"No runtime detector is proposed — the work focuses on measurement and
mitigation rather than detection infrastructure."*

**[Intent-to-Execution Integrity](https://arxiv.org/abs/2605.16976)** (preprint v1, May 2026)
constructs the definitional framework for this property class and *"does not propose a
specific verification implementation."*

Two 2026 papers establish the problem and stop short of the detector.

### 2.3 Every runtime detector in this space is fitted, and the fitting set cannot be certified

**[Real-Time Detection and Repair of LLM Agent Failures](https://arxiv.org/abs/2608.02464)**
(preprint v1, August 2026): echo-state network ensemble with CUSUM alarms, fitted in 1.7 s by
closed-form ridge regression on healthy runs alone, **~200 µs per step**. Threshold is an
empirical quantile of healthy validation episodes:

```
θ = Q₁₋β( { max_t s_t : healthy validation episodes } )
```

| | |
|---|---|
| calibrated on qwen2.5:7b, deployed on llama3.1:8b | **AUROC 0.527** (chance), **healthy false-alarm rate 0.75** |
| same target, recalibrated on itself | AUROC 0.885 |
| organic failure rate in 30 unlabelled episodes at temperature 0.9 | **11/30** |

Stated limitations: *"monitors do not transfer across deployments without recalibration"*;
*"the healthy null is highly specific to the deployment configuration"*; blind to
*"plausible-value corruption — a wrong-but-well-formed number — undetectable from telemetry
by construction."*

**The circularity.** "Healthy" is not defined independently: *"An episode is healthy, or
contains an onset at unknown step τ after which the trajectory distribution shifts and the
run ends in failure."* Healthy = not-failed. Failure is identified by a **study oracle** —
which exists because it is a benchmark, and is the resource absent in deployment. Organic
failures were categorised by hand.

A fitted null inherits an assumption that cannot be checked in the field: **that the fitting
set contains no undetected instances of the event being detected.** Their own 11-in-30 figure
bounds how optimistic that is. The paper does not address minimum healthy-set size,
contamination detection, or protocol when healthy data is unavailable.

This is not a criticism of that work. The field pays this cost collectively, which is why it
reads as a shared limitation rather than a defect — and why it becomes the comparison that
gives this paper a reason to exist.

### 2.4 The gap, stated

> The failure mode is documented and consequential. No runtime detector exists for it. Every
> runtime detector in the adjacent space monitors **behaviour and outputs**, not the return
> path, and sets its threshold by fitting a set that cannot be certified free of the events
> it is meant to detect.

---

## 3. The seam

One seam: the loop's return path, compared **across a turn boundary**. Definition, compared
region and exclusions: **F2**.

The carried-forward region of `context_t` is the span specified to hold `pair_1 … pair_{t−1}`
unchanged. System material, the current prompt, and anything inserted at the current turn lie
outside it.

**Not this seam:** a same-event check comparing the context an application assembled against
the context that reached inference. Identity null, no theorem. Retired (F9).

### 3.1 Why LLM systems are the right empirical domain

The hardest requirement is a **paired interface record** — emitted and delivered, logged
separately on a common alphabet. In most deployed systems that instrumentation does not
exist. In an LLM pipeline it is two log lines, plus one delimiter around the carried-forward
region (F2). This is the argument for the paper's existence and should be made explicitly.

---

## 4. The reading

Estimator: **F3**. Bound: **F4**. Declared parameters: **F5** (K), **F6** (W), **F7** (n_min).

Three things follow that a fitted detector cannot offer:

1. **No healthy corpus.** The null comes from W and K, computed before any data exists.
2. **No recalibration on deployment change.** The band is identical on every model, task and
   vocabulary.
3. **A null class stated in advance**, rather than discovered from what the detector happened
   to miss.

### 4.1 The ladder

```
eventwise fidelity   ⊃   distributional fidelity   ⊃   entropy fidelity
 (blind to nothing)       (blind to changes           (blind to changes
                           preserving the empirical    preserving stream entropy)
                           distribution)
```

Eventwise closure ⟹ distributional closure ⟹ informational closure.

The ladder is presented as **differences in resolution and null class**, not as three
competing detectors. Its necessity comes from Ω's two derived blind spots each having a named
lower rung that covers it:

| Ω is blind to | why | covered by |
|---|---|---|
| removal of content distributionally like what remains, **at any breach size** | Ω sees the records only through their entropies (F3) | distributional rung |
| single-pair loss, at any W and K | excursion ceiling coincides with the band (F4, D2) | eventwise rung |
| entropy-preserving rewriting | H unchanged by construction (D5a) | distributional rung |
| entropy-preserving symbol substitution | H unchanged by construction (D5a) | eventwise rung |

**The paper deliberately includes conditions falling into each null class.** Reporting only
successful detections would be weaker.

### 4.2 Why a diff is not the answer

With both records logged in full, an aligned comparison detects any change — that is the
eventwise rung, the top of the ladder, not a competitor to it. The ladder exists because the
finer rungs cost more, require both records in full, and have no tolerance for specified
transformation. Ω is the rung that survives partial observability and whose tolerance can be
declared before deployment.

---

## 5. What the evidence looks like

Claim status: `02_Claim_Table_v3.md`. Conditions and results: `03_working_log.csv`.

**Two facts to state plainly in the paper rather than let a reviewer find:**

1. **The specificity claim has no held evidence.** The three earlier experiments measured the
   per-turn first difference `H(m_t) − H(m_{t−1})` on a single-stream record where closure
   held by construction. That quantity is retired (F9). T2 is the first evidence for C1.
2. **The derived layer is ahead of the empirical layer.** Five derived or numerically verified
   results; twelve performance claims with no data. Unusual, and better stated than disguised.

Retained as **scope-setting only**, in one short subsection: that Ω does not track turn-level
task advancement (preregistered, n = 192, p = 0.291); that a uniform level shift in message
length does not move the reading; that decoder settings and instructed length do not move it.
All three measured on the retired quantity and marked as such.

---

## 6. Falsification

- Ω exits the band under a manipulation that does not breach the carried-forward region →
  specificity broken.
- Ω stays inside the band under a **distributionally distinct** truncation or compaction at
  realistic magnitudes → useless even if the theorem holds. Similarity-matched breaches are
  predicted null (D4) and do not falsify.
- A predicted null fires → the derived null class is wrong. Worse than a missed detection,
  because it is a claim about the theorem.
- The distributional rung fails to cover Ω's similarity blind spot on real text as it does in
  simulation → the ladder's necessity argument collapses.
- The band at W = 25 is so wide that no realistic breach clears it → quantitative defeat.
  **The only remedy is a smaller window** (F6); coarsening K does not help (F5).

---

## 7. Attack this first

1. **D1's full proof is not written.** The continuity estimate and its conditions must be
   stated.
2. **D6 has no derived null**, and the ladder's necessity argument rests on it.
3. **D4's closed form is owed.** The overlap-1.0 floor under equal weighting is a different
   quantity from the pooled form, and the simulated value is not a derivation.
4. **No breach data exists.** The entire sensitivity arm is unrun.
5. **The signal ratio at W = 25 is extrapolated**, not measured. Confirmed at W = 100 only.
6. **Reframing risk.** The retained scope-setting results were collected as negative results
   about a different quantity. Their role here is narrow and must be stated as such.
7. **Single task, single model pair** until T6.
8. **Ω covers one failure class** where the nearest competitor covers five. The trade must be
   argued, not glossed.
9. **The simulations are a count-vector model, not text.** Zipf draws with a permuted second
   distribution are a caricature of topic change. The structural results should survive; the
   levels will differ on real transcripts.

---

## Appendix — notation

| symbol | meaning |
|---|---|
| `pair_t` | `(p_t, r_t)` — prompt sent, response returned at turn *t* |
| `q_i` | pair token distribution — normalised counts over the declared vocabulary |
| `q̄_Z`, `q̄_S` | emitted and received window distributions, equal-weight mixtures of *W* pair distributions |
| `Ω` | `H(q̄_Z) − H(q̄_S)` |
| `W` | window length in pairs — frozen at 25 (F6) |
| `K` | declared vocabulary size — 32,000 (F5) |
| `n_min` | minimum pair token count for estimator eligibility — 5 (F7) |
| `m` | breach size in pairs |
| `h(·)` | binary entropy function |

## Appendix — cited work

All arXiv preprints, none peer-reviewed as of this date. Cite as preprints.

| id | title |
|---|---|
| [2606.22528](https://arxiv.org/abs/2606.22528) | Governance Decay: How Context Compaction Silently Erases Safety Constraints in Long-Horizon LLM Agents |
| [2605.16976](https://arxiv.org/abs/2605.16976) | Securing LLM Agents Need Intent-to-Execution Integrity |
| [2608.02464](https://arxiv.org/abs/2608.02464) | Real-Time Detection and Repair of LLM Agent Failures |
| [2608.11242](https://arxiv.org/html/2608.11242) | Lost in Compaction: Evaluating Side-Constraint Loss under Context Compaction |
