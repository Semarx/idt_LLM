"""Phase 0 - construction and occupancy screens.

Run before reading any value. Reports, per channel: message counts, token-count
distribution, and distinct tokens against token count. A channel whose distinct
count rises in step with its token count is not yet estimating a distribution.
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def screen(messages_csv):
    M = pd.read_csv(messages_csv)
    print(f"messages {len(M)}   conversations {M.conv_id.nunique()}   "
          f"turns {M.turn.max()} max\n")

    print("=== per channel ===")
    hdr = f"{'channel':10s} {'n':>7s} {'tokens p25/50/75':>20s} {'types/tok':>10s} " \
          f"{'tok/type':>9s} {'corr(types,tok)':>16s}"
    print(hdr)
    for role, g in M.groupby("role"):
        q = np.percentile(g.tokens, [25, 50, 75]).astype(int)
        ratio = (g.types / g.tokens.clip(lower=1)).median()
        occ = (g.tokens / g.types.clip(lower=1)).median()
        rho = spearmanr(g.types, g.tokens).statistic
        print(f"{role:10s} {len(g):7d} {str(tuple(q)):>20s} {ratio:10.3f} "
              f"{occ:9.2f} {rho:16.3f}")

    print("\n=== same, over adjacent token pairs (for later order-based work) ===")
    print(f"{'channel':10s} {'pairs p50':>10s} {'types2/pairs':>13s} {'pairs/type2':>12s}")
    for role, g in M.groupby("role"):
        h = g[g.pairs > 0]
        print(f"{role:10s} {int(h.pairs.median()):10d} "
              f"{(h.types_2/h.pairs).median():13.3f} "
              f"{(h.pairs/h.types_2.clip(lower=1)).median():12.2f}")

    print("\n=== entropy by channel (bits) ===")
    for role, g in M.groupby("role"):
        print(f"  {role:10s} H  mean {g.H.mean():5.2f}  median {g.H.median():5.2f}  "
              f"sd {g.H.std():5.2f}   |   H_2 median {g.H_2.median():5.2f}")

    print("\nReading: 'tok/type' is samples per occupied cell. Values near 1 mean each")
    print("token appears about once, so the entropy is largely reporting message size.")
    return M


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", default="Data/messages.csv")
    screen(ap.parse_args().messages)
