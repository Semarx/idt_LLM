"""Phase 1 - the two Omega series.

Omega_model = H(prompt t+1)   - H(prompt t)     the user's stream, as the model faces it
Omega_user  = H(response t+1) - H(response t)   the model's stream, as the user faces it

Both are read from the same message table. Writes Data/omega.csv, one row per turn.
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def build_omega(messages_csv, out_csv="Data/omega.csv"):
    M = pd.read_csv(messages_csv).sort_values(["conv_id", "seq"])
    P = M[M.role == "prompt"][["conv_id", "turn", "H", "H_2", "tokens"]]
    R = M[M.role == "response"][["conv_id", "turn", "H", "H_2", "tokens"]]
    P = P.rename(columns={"H": "H_p", "H_2": "H2_p", "tokens": "tok_p"})
    R = R.rename(columns={"H": "H_r", "H_2": "H2_r", "tokens": "tok_r"})
    T = P.merge(R, on=["conv_id", "turn"]).sort_values(["conv_id", "turn"])

    g = T.groupby("conv_id")
    T["Om_model"] = g.H_p.shift(-1) - T.H_p          # next prompt minus this prompt
    T["Om_user"] = g.H_r.shift(-1) - T.H_r           # next response minus this response
    T["dOm"] = T.Om_model - T.Om_user
    T["d_tok_p"] = g.tok_p.shift(-1) - T.tok_p       # matched covariate for Om_model
    T["d_tok_r"] = g.tok_r.shift(-1) - T.tok_r
    T = T.dropna(subset=["Om_model", "Om_user"])
    T.to_csv(out_csv, index=False)
    return T


def report(T):
    print(f"turns with both quantities defined: {len(T)}   "
          f"conversations: {T.conv_id.nunique()}\n")

    print("=== distributions (bits) ===")
    print(f"{'quantity':10s} {'mean':>8s} {'median':>8s} {'sd':>8s} {'p5':>8s} {'p95':>8s}")
    for c in ["Om_model", "Om_user", "dOm"]:
        v = T[c]
        print(f"{c:10s} {v.mean():8.4f} {v.median():8.4f} {v.std():8.4f} "
              f"{v.quantile(.05):8.3f} {v.quantile(.95):8.3f}")

    print("\n=== within-conversation dispersion (sd of the per-turn series) ===")
    for c in ["Om_model", "Om_user", "dOm"]:
        s = T.groupby("conv_id")[c].std()
        print(f"  {c:10s} median sd across conversations {s.median():.4f}")

    print("\n=== are the two series related? (per conversation, >=30 turns) ===")
    v = []
    for cid, g in T.groupby("conv_id"):
        if len(g) >= 30:
            r = spearmanr(g.Om_model, g.Om_user).statistic
            if np.isfinite(r):
                v.append(r)
    v = np.array(v)
    print(f"  Spearman(Om_model, Om_user): mean {v.mean():+.3f}  "
          f"median {np.median(v):+.3f}  range {v.min():+.3f} to {v.max():+.3f}  "
          f"(n={len(v)} conversations)")

    print("\n=== telescoping check: |mean Omega| over a window should fall as 1/W ===")
    print(f"{'W':>6s} {'|mean Om_model|':>16s} {'|mean Om_user|':>15s} {'1/W':>9s}")
    for W in [5, 10, 20, 40, 80]:
        a, b = [], []
        for cid, g in T.groupby("conv_id"):
            x, y = g.Om_model.values, g.Om_user.values
            for i in range(0, len(x) - W + 1, W):
                a.append(abs(x[i:i+W].mean()))
                b.append(abs(y[i:i+W].mean()))
        if a:
            print(f"{W:6d} {np.mean(a):16.4f} {np.mean(b):15.4f} {1.0/W:9.4f}")
    print("\n  A log-log slope near -1 confirms the construction telescopes as expected.")
    for name, col in [("Om_model", "Om_model"), ("Om_user", "Om_user")]:
        Ws, vals = [], []
        for W in [5, 10, 20, 40, 80]:
            a = []
            for cid, g in T.groupby("conv_id"):
                x = g[col].values
                for i in range(0, len(x) - W + 1, W):
                    a.append(abs(x[i:i+W].mean()))
            if a:
                Ws.append(W); vals.append(np.mean(a))
        sl = np.polyfit(np.log(Ws), np.log(vals), 1)[0]
        print(f"    {name}: fitted slope {sl:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", default="Data/messages.csv")
    ap.add_argument("--out", default="Data/omega.csv")
    a = ap.parse_args()
    report(build_omega(a.messages, a.out))
