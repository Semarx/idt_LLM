"""
verify_band_scaling.py — reproduces the three analytic checks in
Seam_Integrity_Spine_v3.md §6.

    1. Is K a lever?            band and excursion both carry log2(K); the ratio is flat.
    2. The similarity limit.    Omega goes blind when removed content matches retained
                                content, regardless of breach size. JS does not.
    3. Single-turn resolution.  For m = 1 the excursion ceiling equals the band exactly.

Nothing here uses experimental data. It is a simulation of the count-vector algebra, run
to confirm the analytic statements before any generation is spent on them.

    python src/verify_band_scaling.py
"""
import numpy as np
from math import log2

TOK_PER_TURN = 120
N_TRIALS = 12


# ---------------------------------------------------------------------------

def h(p):
    """Binary entropy, bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def band(W, K):
    """Proposition 1 edge band: (1/W)log2(K-1) + h(1/W)."""
    return (1 / W) * log2(K - 1) + h(1 / W)


def excursion_ceiling(W, K, m):
    """Max |dH| attainable at total variation m/W, same continuity estimate."""
    d = m / W
    return d * log2(K - 1) + h(d)


def H(counts):
    c = counts[counts > 0]
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def JS(a, b):
    pa, pb = a / a.sum(), b / b.sum()
    m = 0.5 * (pa + pb)
    def kl(p, q):
        s = p > 0
        return float((p[s] * np.log2(p[s] / q[s])).sum())
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def zipf(K, s=1.1):
    w = 1.0 / np.power(np.arange(1, K + 1), s)
    return w / w.sum()


def one_trial(W, K, m, overlap, rng):
    """Remove m of W turns. Retained turns ~ P_A; removed turns ~
    overlap*P_A + (1-overlap)*P_B, so overlap=1 means the removed content is
    distributionally identical to what remains."""
    pA = zipf(K)
    pB = pA[rng.permutation(K)]
    pRem = overlap * pA + (1 - overlap) * pB

    kept = np.zeros(K)
    for _ in range(W - m):
        kept += np.bincount(rng.choice(K, TOK_PER_TURN, p=pA), minlength=K)
    full = kept.copy()
    for _ in range(m):
        full += np.bincount(rng.choice(K, TOK_PER_TURN, p=pRem), minlength=K)

    return abs(H(full) - H(kept)), JS(full, kept)


def mean_trial(W, K, m, overlap, rng, n=N_TRIALS):
    r = [one_trial(W, K, m, overlap, rng) for _ in range(n)]
    return float(np.mean([x[0] for x in r])), float(np.mean([x[1] for x in r]))


# ---------------------------------------------------------------------------

def check_1_alphabet(rng):
    print("=== 1. is K a lever? ===")
    print("    W=100, m=20, overlap=0.0. K swept over a 1000x range.\n")
    print(f"    {'K':>8} {'band':>8} {'|Omega|':>9} {'ratio':>7}")
    for K in (128, 1280, 12800, 128000):
        om, _ = mean_trial(100, K, 20, 0.0, rng)
        b = band(100, K)
        print(f"    {K:8d} {b:8.4f} {om:9.4f} {om/b:7.2f}")
    print("\n    Band and excursion both carry log2(K); the ratio is flat.")
    print("    Coarsening the alphabet is not a remedy. W is the only lever.\n")


def check_2_similarity(rng):
    W, K, m = 100, 12800, 40
    b = band(W, K)
    js_edge = 0.5 / W                      # crude order-of-magnitude reference
    print("=== 2. the similarity limit ===")
    print(f"    W={W}, K={K}, m={m} (a {m/W:.0%} truncation). "
          f"Omega band = {b:.4f}.\n")
    print(f"    {'overlap':>8} {'|Omega|':>9} {'ratio':>7} {'Omega':>12} "
          f"{'JS':>9} {'JS/edge':>8} {'JS':>12}")
    for ov in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        om, js = mean_trial(W, K, m, ov, rng)
        print(f"    {ov:8.2f} {om:9.4f} {om/b:7.2f} "
              f"{'BLIND' if om < b else 'detects':>12} "
              f"{js:9.5f} {js/js_edge:8.2f} "
              f"{'blind' if js < js_edge else 'detects':>12}")
    print("\n    Omega goes blind when removed content matches retained content,")
    print("    regardless of how much is removed. JS does not: it separates on")
    print("    effective sample size even at identical underlying distributions.")
    print("    This is what makes the distributional rung load-bearing.\n")


def check_3_single_turn(rng):
    print("=== 3. single-turn resolution limit ===")
    print("    For m=1 the total variation is the same 1/W the closure endpoint")
    print("    produces, so the excursion ceiling is the band, identically.\n")
    print(f"    {'W':>5} {'K':>7} {'band':>8} {'ceiling':>8} {'ratio':>7} "
          f"{'observed':>9} {'obs/band':>9}")
    for W in (25, 50, 100, 200):
        for K in (256, 32000):
            b, c = band(W, K), excursion_ceiling(W, K, 1)
            om, _ = mean_trial(W, K, 1, 0.0, rng)
            print(f"    {W:5d} {K:7d} {b:8.4f} {c:8.4f} {c/b:7.3f} "
                  f"{om:9.4f} {om/b:9.2f}")
    print("\n    Ratio is exactly 1.000 by construction. No W and no K change it.")
    print("    Single-turn loss belongs to the eventwise rung, analytically.\n")


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    check_1_alphabet(rng)
    check_2_similarity(rng)
    check_3_single_turn(rng)
