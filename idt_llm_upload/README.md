# Toward Measuring Structural Drift in LLM Communication Loops

Code and derived results for the manuscript. Supersedes the earlier contents of this
repository, which belonged to a withdrawn formulation and are not part of this study.

Two quantities are computed:

- **Omega_T** — signed entropy balance across a declared seam (communication closure).
- **epsilon_A** — normalized conditional contribution, I(A;S'|S) / H(S'|S).

## Corpora

| | Description | Source data in this repository |
|---|---|---|
| **C1** | 58 ordinary human–LLM conversations | **Not included** — personal correspondence. Derived results are included in full. |
| **C2** | 8 controlled LLM–LLM runs | Included in `data/c2/`. |
| **C3** | 2,171 human–human PhotoBook dialogues | **Text not redistributed.** `data/c3/c3_manifest.csv` records every game with its turn count and exclusion reason; `scripts/prep_c3.py` rebuilds the working files from the public corpus. |

## Layout

```
src/         estimator and closure implementations, C2 generation pipeline
scripts/     one script per analysis stage and per figure
data/        C2 source, C3 manifest, C1 placeholder
results/     every derived value cited in the manuscript
figures/     figures as published
docs/        frozen decisions, proofs, claim table
```

## Declared parameters

Closure uses the Llama-2 tokenizer with K = 32,000 as a fixed observational alphabet.
Contribution uses `all-MiniLM-L6-v2`, mean-pooled and L2-normalized, `max_length=256`,
with k-means at `random_state=0`, `n_init=10`. Symbol count k is 2 for records of
25–149 turns, 3 for 150–399, 4 for 400 and above. The eligibility floor is 25 turns.

Randomization is deterministic: permutation seeds are derived by MD5 hash from record
identifiers, so every run reproduces exactly.

## Data availability

C1 source conversations are withheld as personal correspondence. All C1 derived
results are included and contain no message text; record identifiers are opaque
hashes. Every C1 number in the manuscript can be verified from `results/`, but not
regenerated from source.

## Reproduction

See `REPRODUCE.md` for the display-item to script to data map.

## Title of record

Hafez, W., Wei, C. & Nazeri, A. *Toward Measuring Structural Drift in LLM Communication Loops.*
Under review, Scientific Reports.
