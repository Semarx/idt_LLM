"""
verify_estimator.py — analytic checks behind the seam-integrity paper.

Uses NO experimental data. Everything here is a simulation of the estimator algebra,
run to confirm the analytic statements before generation is spent on them.

    A. why equal pair weighting is required   (the pooled estimator breaks TV <= 1/W)
    B. D3 — K is not a design lever
    C. D4 — the similarity limit
    D. D2 — the single-pair resolution limit
    E. n_min — estimator-reliability calibration
    F. W — noise/band scaling, which bounds useful window length

Estimator under test (equal pair weighting):

    q_i(k)  = count of token k in pair i / total tokens in pair i
    qbar_Z  = (1/W) sum_{i=t-W+1..t}   q_i        emitted window distribution
    qbar_S  = (1/W) sum_{i=t-W..t-1}   q_i        received window distribution
    Omega   = H(qbar_Z) - H(qbar_S)

    python src/verify_estimator.py
"""
import numpy as np
from math import log2

K_DEFAULT = 32000
TRIALS = 300
TOL = 0.05


def h(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def band(W, K):
    """Proposition 1 edge band."""
    return (1 / W) * log2(K - 1) + h(1 / W)


def H(v):
    v = v[v > 0]
    return float(-(v * np.log2(v)).sum())


def TV(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def zipf(K, s=1.1):
    w = 1.0 / np.power(np.arange(1, K + 1), s)
    return w / w.sum()


def q_pair(tokens, K):
    """Normalised token distribution of one pair."""
    return np.bincount(tokens, minlength=K) / len(tokens)


def window_dist(pairs):
    """Equal-weight mixture of per-pair distributions. Never pool raw tokens."""
    return np.mean(pairs, axis=0)


# ---------------------------------------------------------------------------

def check_A_weighting(rng):
    print("=== A. why equal pair weighting is required ===")
    K, W = 2000, 25
    p = zipf(K)
    lens = [30] * (W - 1) + [3000]        # one long pair among short ones
    pairs = [rng.choice(K, n, p=p) for n in lens]
    entering = rng.choice(K, 30, p=p)

    def pooled(ps):
        c = np.zeros(K)
        for x in ps:
            c += np.bincount(x, minlength=K)
        return c / c.sum()

    eq = lambda ps: window_dist([q_pair(x, K) for x in ps])
    shifted = [entering] + pairs[:-1]

    print(f"    W={W}, K={K}, one 3000-token pair among 30-token pairs")
    print(f"    1/W = {1/W:.4f}")
    print(f"    pooled tokens      TV = {TV(pooled(pairs), pooled(shifted)):.4f}"
          f"   {'VIOLATES the bound' if TV(pooled(pairs), pooled(shifted)) > 1/W else 'ok'}")
    print(f"    equal-weight pairs TV = {TV(eq(pairs), eq(shifted)):.4f}"
          f"   {'VIOLATES the bound' if TV(eq(pairs), eq(shifted)) > 1/W else 'ok'}")
    print("\n    qbar_Z - qbar_S = (q_t - q_{t-W})/W, so TV <= 1/W with no assumption")
    print("    about pair lengths. Pooling reintroduces one.\n")


def breach_omega(W, K, m, overlap, rng, lo=60, hi=240):
    """Remove m of W pairs. Retained ~ P_A; removed ~ overlap*P_A + (1-overlap)*P_B."""
    pA = zipf(K)
    pB = pA[rng.permutation(K)]
    pR = overlap * pA + (1 - overlap) * pB
    kept = [q_pair(rng.choice(K, rng.integers(lo, hi), p=pA), K) for _ in range(W - m)]
    rem = [q_pair(rng.choice(K, rng.integers(lo, hi), p=pR), K) for _ in range(m)]
    return abs(H(window_dist(kept + rem)) - H(window_dist(kept)))


def check_B_alphabet(rng):
    print("=== B. D3 — is K a design lever? ===")
    print("    W=100, m=20, overlap=0.0. K swept over a 1000x range.\n")
    print(f"    {'K':>8} {'band':>8} {'|Omega|':>9} {'ratio':>7}")
    for K in (128, 1280, 12800, 128000):
        v = np.mean([breach_omega(100, K, 20, 0.0, rng) for _ in range(12)])
        b = band(100, K)
        print(f"    {K:8d} {b:8.4f} {v:9.4f} {v/b:7.2f}")
    print("\n    Flat. Band and excursion both carry log2(K) and it cancels.")
    print("    The tokenizer is a free declaration, not a tuning knob.\n")


def check_C_similarity(rng):
    W, K, m = 100, 12800, 40
    b = band(W, K)
    print("=== C. D4 — the similarity limit ===")
    print(f"    W={W}, K={K}, m={m} (a {m/W:.0%} removal). band = {b:.4f}\n")
    print(f"    {'overlap':>8} {'|Omega|':>9} {'ratio':>7}   verdict")
    for ov in (0.0, 0.5, 0.75, 0.9, 1.0):
        v = np.mean([breach_omega(W, K, m, ov, rng) for _ in range(12)])
        print(f"    {ov:8.2f} {v:9.4f} {v/b:7.2f}   "
              f"{'BLIND' if v < b else 'detects'}")
    print("\n    A similarity limit, not a size limit: breach magnitude is fixed")
    print("    across the whole table. Already marginal by overlap 0.90.\n")


def check_D_single_pair(rng):
    print("=== D. D2 — the single-pair resolution limit ===")
    print("    A one-pair breach displaces qbar_S by (qbar_others - q_j)/W, giving")
    print("    TV <= 1/W — the same total variation closure itself produces.\n")
    print(f"    {'W':>5} {'K':>7} {'band':>8} {'observed':>9} {'ratio':>7}")
    for W in (25, 100, 200):
        K = K_DEFAULT
        v = np.mean([breach_omega(W, K, 1, 0.0, rng) for _ in range(12)])
        print(f"    {W:5d} {K:7d} {band(W,K):8.4f} {v:9.4f} {v/band(W,K):7.2f}")
    print("\n    The excursion ceiling equals the band identically, at any W, any K,")
    print("    and any pair-length distribution. Eventwise rung's business.\n")


def noise_omega(W, K, n, p, rng):
    """Pure estimator noise: two INDEPENDENT realisations of the same W pair
    distributions. True Omega = 0; anything observed is finite-sample noise."""
    idx = rng.choice(K, size=(2, W, n), p=p)
    out = []
    for s in range(2):
        acc = np.zeros(K)
        np.add.at(acc, idx[s].ravel(), 1.0 / n)
        out.append(acc / W)
    return abs(H(out[0]) - H(out[1]))


def check_E_nmin():
    print("=== E. n_min — estimator-reliability calibration ===")
    print("    Two independent realisations of identical pair distributions.")
    print("    True Omega = 0 by construction. No breach, no closure structure,")
    print(f"    no detection performance. Criterion: false-excursion rate <= {TOL:.0%}.")
    print(f"    K={K_DEFAULT}, {TRIALS} trials per cell.\n")
    ns = (5, 10, 15, 20, 30, 40, 60)
    p = zipf(K_DEFAULT)
    print(f"    {'W':>5} {'band':>8} | " + " ".join(f"{n:>6}" for n in ns) + "   n_min")
    rows = {}
    for W in (25, 50, 100, 200):
        b = band(W, K_DEFAULT)
        rng = np.random.default_rng(101)
        fes, means, nmin = [], [], None
        for n in ns:
            v = np.array([noise_omega(W, K_DEFAULT, n, p, rng) for _ in range(TRIALS)])
            fe = float(np.mean(v > b))
            fes.append(fe); means.append(v.mean())
            if nmin is None and fe <= TOL:
                nmin = n
        rows[W] = (b, means)
        print(f"    {W:5d} {b:8.4f} | " + " ".join(f"{f:6.3f}" for f in fes)
              + f"   {nmin}")
    return rows, ns


def check_F_window(rows, ns):
    print("\n=== F. W — noise/band scaling ===")
    print("    The band tightens as 1/W. Estimator noise tightens only as 1/sqrt(W).")
    print("    So the ratio GROWS as sqrt(W): longer windows are worse.\n")
    print(f"    {'W':>5} | " + " ".join(f"{n:>6}" for n in ns))
    for W, (b, means) in rows.items():
        print(f"    {W:5d} | " + " ".join(f"{m/b:6.2f}" for m in means))
    print("\n    An m-pair breach gives a signal/band ratio depending on m alone,")
    print("    not on W (check B). Larger W therefore buys nothing in signal and")
    print("    costs sqrt(W) in noise. W should be as SMALL as the application allows.\n")


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    check_A_weighting(rng)
    check_B_alphabet(rng)
    check_C_similarity(rng)
    check_D_single_pair(rng)
    rows, ns = check_E_nmin()
    check_F_window(rows, ns)
