"""
omega_advancement_test_v1  -- preregistered analysis

Reads the generated turns and executes the frozen plan. Nothing here is chosen
after seeing data.

    Omega_prompt_t   = H_prompt_t   - H_prompt_(t-1)
    Omega_response_t = H_response_t - H_response_(t-1)
    dOmega_t         = Omega_response_t - Omega_prompt_t
    abs_dOmega_t     = |dOmega_t|

    D = mean(abs_dOmega | advance=1) - mean(abs_dOmega | advance=0)
    prediction: D < 0, one-sided

Null: the Omega sequence is held fixed and only the advance/stall labels are
permuted, within each conversation, preserving that conversation's counts.
10,000 permutations. Omega is never shuffled.

Order of operations is enforced by the script: the synthetic-noise validation
runs and is written to disk BEFORE the real statistic is computed.

Exclusions, frozen:
    - turn 1 of every conversation
    - any turn hitting num_predict, and the turn after it
    - any turn flagged by the context guard
No outlier trimming. No conversation dropped on the basis of its Omega values.

Frozen handling rules:
    A  flagged conversations (<3 advancing or <3 stalled usable turns) are
       INCLUDED in the primary pooled D. Excluding them is reported only as a
       clearly secondary robustness line.
    B  prompts are never edited; n_question_marks is a diagnostic only.
    C  turn-index diagnostics are computed but kept out of the primary file.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msg_entropy import Tokenizer, message_stats

EXPERIMENT_ID = "omega_advancement_test_v1"
N_PERM = 10000
SEED_RNG = 20260907


# ---------------------------------------------------------------------------
# statistic and null
# ---------------------------------------------------------------------------

def D_stat(abs_d, advance):
    a = abs_d[advance == 1]
    b = abs_d[advance == 0]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return a.mean() - b.mean()


def within_conv_permute(advance, conv_idx, rng):
    """Shuffle labels inside each conversation; counts per conversation are
    preserved exactly."""
    lab = advance.copy()
    for m in conv_idx:
        lab[m] = rng.permutation(lab[m])
    return lab


def permute_test(abs_d, advance, conv_idx, rng, n=N_PERM, stat=D_stat):
    obs = stat(abs_d, advance)
    null = np.empty(n)
    for i in range(n):
        null[i] = stat(abs_d, within_conv_permute(advance, conv_idx, rng))
    p = (np.sum(null <= obs) + 1) / (n + 1)          # one-sided, D < 0
    return obs, p, null


def ols_advance_coef(abs_d, advance, dlp, dlr):
    """Coefficient on `advance` in abs_dOmega ~ advance + dlog_p + dlog_r."""
    X = np.column_stack([np.ones_like(abs_d), advance.astype(float), dlp, dlr])
    beta, *_ = np.linalg.lstsq(X, abs_d, rcond=None)
    return beta[1]


# ---------------------------------------------------------------------------
# the standing rule: validate the null on pure noise, first, and record it
# ---------------------------------------------------------------------------

def validate_null(conv_sizes, adv_rate, rng, n_trials=300, n_perm=400):
    """Synthetic H series -> first difference -> synthetic Omega -> random
    labels -> the identical permutation procedure. Expect mean p near 0.5 and
    a false-positive rate near 0.05."""
    ps = []
    for _ in range(n_trials):
        abs_d, adv, idx, off = [], [], [], 0
        for n in conv_sizes:
            Hp = rng.normal(size=n + 1)
            Hr = rng.normal(size=n + 1)
            d = np.abs(np.diff(Hr) - np.diff(Hp))
            a = (rng.random(n) < adv_rate).astype(int)
            abs_d.append(d); adv.append(a)
            idx.append(np.arange(off, off + n)); off += n
        abs_d = np.concatenate(abs_d); adv = np.concatenate(adv)
        if adv.sum() == 0 or adv.sum() == len(adv):
            continue
        _, p, _ = permute_test(abs_d, adv, idx, rng, n=n_perm)
        ps.append(p)
    ps = np.array(ps)
    return {"n_trials": len(ps), "mean_p": float(ps.mean()),
            "false_positive_rate_0.05": float(np.mean(ps < 0.05)),
            "expected_mean_p": 0.5, "expected_fpr": 0.05,
            "behaving": bool(0.40 < ps.mean() < 0.60
                             and np.mean(ps < 0.05) < 0.12)}


# ---------------------------------------------------------------------------

def build(turns_csv, tokenizer):
    tok = Tokenizer(tokenizer)
    T = pd.read_csv(turns_csv).sort_values(["run_id", "turn"]).reset_index(drop=True)
    T["tok_p"] = [len(tok.ids(x)) for x in T.prompt]
    T["tok_r"] = [len(tok.ids(x)) for x in T.response]
    T["H_prompt"]   = [message_stats(tok.ids(x))["H"] for x in T.prompt]
    T["H_response"] = [message_stats(tok.ids(x))["H"] for x in T.response]

    g = T.groupby("run_id", sort=False)
    T["Omega_prompt"]   = T.H_prompt   - g.H_prompt.shift(1)
    T["Omega_response"] = T.H_response - g.H_response.shift(1)
    T["dOmega"]     = T.Omega_response - T.Omega_prompt
    T["abs_dOmega"] = T.dOmega.abs()
    T["dlog_tokens_prompt"]   = (np.log(T.tok_p) - np.log(g.tok_p.shift(1))).abs()
    T["dlog_tokens_response"] = (np.log(T.tok_r) - np.log(g.tok_r.shift(1))).abs()

    # frozen exclusions
    T["excl_turn1"]   = (T.turn == 1).astype(int)
    T["excl_ceiling"] = (T.ceiling_hit.fillna(0).astype(int) > 0).astype(int)
    T["excl_after_ceiling"] = (g.ceiling_hit.shift(1).fillna(0).astype(int) > 0).astype(int)
    T["excl_context"] = (T.context_flag.fillna(0).astype(int) > 0).astype(int)
    T["usable"] = ((T.excl_turn1 + T.excl_ceiling + T.excl_after_ceiling
                    + T.excl_context) == 0) & T.abs_dOmega.notna()
    return T


def conv_index(U):
    return [np.where(U.run_id.values == c)[0] for c in U.run_id.unique()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default="Data/advancement/omega_advancement_test_v1_turns.csv")
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.dirname(os.path.abspath(a.turns))
    rng = np.random.default_rng(SEED_RNG)

    T = build(a.turns, a.tokenizer)
    T.to_csv(os.path.join(out, f"{EXPERIMENT_ID}_turnlevel.csv"), index=False)
    U = T[T.usable].reset_index(drop=True)
    idx = conv_index(U)

    # ---- step 1: validate the null, write it to disk, THEN look at data ----
    sizes = [len(i) for i in idx]
    rate = float(U.advance.mean())
    val = validate_null(sizes, rate, np.random.default_rng(SEED_RNG + 1))
    with open(os.path.join(out, f"{EXPERIMENT_ID}_null_validation.json"), "w") as f:
        json.dump(val, f, indent=2)
    print("=== null validation (recorded before the real result) ===")
    print(f"  trials {val['n_trials']}   mean p {val['mean_p']:.3f} "
          f"(expect ~0.50)   FPR@0.05 {val['false_positive_rate_0.05']:.3f} "
          f"(expect ~0.05)")
    print(f"  behaving as expected: {val['behaving']}")
    if not val["behaving"]:
        print("  NOTE: the machinery is misbehaving on noise. Everything below "
              "is unreadable until this is resolved.")

    # ---- step 2: the preregistered primary test ----------------------------
    abs_d = U.abs_dOmega.values
    adv   = U.advance.values.astype(int)
    D_obs, p_perm, _ = permute_test(abs_d, adv, idx, rng)

    dlp, dlr = U.dlog_tokens_prompt.values, U.dlog_tokens_response.values
    r_p = float(np.corrcoef(abs_d, dlp)[0, 1])
    r_r = float(np.corrcoef(abs_d, dlr)[0, 1])

    coef_obs = ols_advance_coef(abs_d, adv, dlp, dlr)
    null_c = np.empty(N_PERM)
    rng2 = np.random.default_rng(SEED_RNG + 2)
    for i in range(N_PERM):
        null_c[i] = ols_advance_coef(abs_d, within_conv_permute(adv, idx, rng2),
                                     dlp, dlr)
    p_coef = (np.sum(null_c <= coef_obs) + 1) / (N_PERM + 1)

    # ---- conversation flags and the secondary robustness line --------------
    flags = []
    for c in U.run_id.unique():
        s = U[U.run_id == c]
        flags.append({"run_id": c, "usable_turns": len(s),
                      "advancing": int(s.advance.sum()),
                      "stalled": int((s.advance == 0).sum()),
                      "insufficient": bool(s.advance.sum() < 3
                                           or (s.advance == 0).sum() < 3)})
    F = pd.DataFrame(flags)
    keep = set(F[~F.insufficient].run_id)
    if len(keep) and keep != set(F.run_id):
        V = U[U.run_id.isin(keep)].reset_index(drop=True)
        D_sec, p_sec, _ = permute_test(V.abs_dOmega.values,
                                       V.advance.values.astype(int),
                                       conv_index(V),
                                       np.random.default_rng(SEED_RNG + 3))
        sec = f"D = {D_sec:+.4f}, p = {p_sec:.4f}, {len(V)} turns, " \
              f"{len(keep)} conversations"
    else:
        sec = "not applicable (no conversation was flagged, or all were)"

    # ---- diagnostics, kept out of the primary file (rule C) ----------------
    diag = U.groupby("turn").agg(n=("advance", "size"),
                                 advance_rate=("advance", "mean"),
                                 mean_abs_dOmega=("abs_dOmega", "mean"))
    diag.to_csv(os.path.join(out, f"{EXPERIMENT_ID}_diagnostics_by_turn.csv"))
    r_turn = float(np.corrcoef(U.turn.values, abs_d)[0, 1])
    r_turn_adv = float(np.corrcoef(U.turn.values, adv)[0, 1])
    F.to_csv(os.path.join(out, f"{EXPERIMENT_ID}_conversation_flags.csv"), index=False)
    with open(os.path.join(out, f"{EXPERIMENT_ID}_diagnostics.txt"), "w") as f:
        f.write(f"corr(turn index, abs_dOmega) = {r_turn:+.4f}\n")
        f.write(f"corr(turn index, advance)    = {r_turn_adv:+.4f}\n")
        f.write(f"single-question compliance   = "
                f"{T.single_question_compliant.mean():.3f}\n")
        f.write(f"mean n_question_marks        = "
                f"{T.n_question_marks.mean():.2f}\n\n")
        f.write(diag.to_string())

    # ---- the primary results file -----------------------------------------
    ma = abs_d[adv == 1].mean(); ms = abs_d[adv == 0].mean()
    lines = [
        f"{EXPERIMENT_ID} -- preregistered primary results",
        "=" * 58, "",
        "HYPOTHESIS   |dOmega| is smaller on advancing turns than on stalled turns",
        "STATISTIC    D = mean(|dOmega| | advance=1) - mean(|dOmega| | advance=0)",
        "PREDICTION   D < 0   (one-sided)",
        "",
        "--- null validation (run and recorded before the result) ---",
        f"  trials                    {val['n_trials']}",
        f"  mean p on pure noise      {val['mean_p']:.4f}   (expect ~0.50)",
        f"  false-positive rate 0.05  {val['false_positive_rate_0.05']:.4f}   (expect ~0.05)",
        f"  behaving as expected      {val['behaving']}",
        "",
        "--- counts ---",
        f"  conversations             {U.run_id.nunique()}",
        f"  usable turns              {len(U)}",
        f"  advancing turns           {int(adv.sum())}",
        f"  stalled turns             {int((adv == 0).sum())}",
        "",
        "--- exclusions ---",
        f"  turn 1                    {int(T.excl_turn1.sum())}",
        f"  hit num_predict           {int(T.excl_ceiling.sum())}",
        f"  turn after a ceiling hit  {int(T.excl_after_ceiling.sum())}",
        f"  context guard             {int(T.excl_context.sum())}",
        f"  total rows                {len(T)}",
        "",
        "--- primary result ---",
        f"  mean |dOmega| advancing   {ma:.4f}",
        f"  mean |dOmega| stalled     {ms:.4f}",
        f"  D                         {D_obs:+.4f}",
        f"  permutation p (one-sided) {p_perm:.4f}   [{N_PERM} within-conversation permutations]",
        "",
        "--- length control ---",
        f"  corr(|dOmega|, dlog_tokens_prompt)    {r_p:+.4f}",
        f"  corr(|dOmega|, dlog_tokens_response)  {r_r:+.4f}",
        f"  advance coefficient, length-controlled {coef_obs:+.4f}",
        f"  permutation p (one-sided)              {p_coef:.4f}",
        "",
        "--- secondary, not the preregistered test ---",
        f"  excluding flagged conversations: {sec}",
        "",
        "Omega was never shuffled. Only advance/stall labels were permuted,",
        "within conversations, preserving each conversation's counts.",
    ]
    txt = "\n".join(lines)
    with open(os.path.join(out, f"{EXPERIMENT_ID}_primary_results.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print(f"\nwrote {out}/{EXPERIMENT_ID}_primary_results.txt")


if __name__ == "__main__":
    main()
