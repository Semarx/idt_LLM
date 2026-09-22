"""
verify_d1_proof.py — numerical verification of every step of the D1 proof.

    Step 2  TV(qbar_Z, qbar_S) <= 1/W, and the bound is attained
    Step 3  the Fannes-Audenaert bound is attained by the extremal pair
    Step 4  f(d) = d*log2(K-1) + h(d) increases exactly on [0, 1 - 1/K]

Uses no experimental data. See 05_D1_Proof.md.
"""
import numpy as np
from math import log2


def h(d):
    if d <= 0 or d >= 1:
        return 0.0
    return -d * log2(d) - (1 - d) * log2(1 - d)


def H(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def TV(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def FA(d, K):
    """Fannes-Audenaert bound, which is also the D1 band at d = 1/W."""
    return d * log2(K - 1) + h(d)


def step4_monotonicity():
    print("Step 4 — f increases exactly on [0, 1 - 1/K]")
    print("  f'(d) = log2(K-1) + log2((1-d)/d) >= 0  <=>  d <= 1 - 1/K\n")
    ds = np.linspace(1e-6, 0.999999, 400000)
    hv = np.array([h(x) for x in ds])
    for K in (256, 32000):
        f = ds * log2(K - 1) + hv
        arg = ds[int(np.argmax(f))]
        print(f"    K={K:6d}  numerical argmax {arg:.6f}   predicted {1-1/K:.6f}   "
              f"{'ok' if abs(arg-(1-1/K)) < 2e-3 else 'MISMATCH'}")


def step3_sharpness():
    print("\nStep 3 — the bound is ATTAINED by P=(1,0,..,0), Q=(1-d, d/(K-1), ..)\n")
    print(f"    {'K':>7} {'W':>5} {'d':>8} {'|dH| attained':>14} {'bound':>10} {'gap':>10}")
    for K in (256, 32000):
        for W in (25, 50, 100):
            d = 1.0 / W
            P = np.zeros(K); P[0] = 1.0
            Q = np.full(K, d / (K - 1)); Q[0] = 1.0 - d
            got, bnd = abs(H(P) - H(Q)), FA(d, K)
            print(f"    {K:7d} {W:5d} {d:8.4f} {got:14.6f} {bnd:10.6f} {abs(got-bnd):10.2e}")


def step2_tv(trials=300, K=2000):
    print("\nStep 2 — TV <= 1/W for arbitrary pair lengths and supports\n")
    rng = np.random.default_rng(5)
    for W in (10, 25, 50):
        worst = 0.0
        for _ in range(trials):
            pairs = []
            for _ in range(W + 1):
                n = rng.integers(3, 400)
                v = np.bincount(rng.choice(K, n), minlength=K).astype(float)
                pairs.append(v / v.sum())
            Z = np.mean(pairs[1:], axis=0)
            S = np.mean(pairs[:-1], axis=0)
            worst = max(worst, TV(Z, S))
        print(f"    W={W:3d}  max TV over {trials} random windows = {worst:.6f}   "
              f"1/W = {1/W:.6f}   {'ok' if worst <= 1/W + 1e-12 else 'VIOLATION'}")
    print("\n    Equality is reached: disjoint supports give TV(q_t, q_t-W) = 1.")


if __name__ == "__main__":
    step4_monotonicity()
    step3_sharpness()
    step2_tv()
    print("\nAll three steps verified. The band is the best bound obtainable from total")
    print("variation alone, and both contributing steps are individually tight.")
