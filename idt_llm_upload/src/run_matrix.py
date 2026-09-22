"""Run the Experiment 1 matrix: 3 seeds x 4 perturbation fractions = 12 runs.

Each run is a separate invocation of generate.py with overrides, so every run
writes its own transcript and manifest exactly as a single run does. Afterwards
one summary row per run is written.

    python src/run_matrix.py
    python src/run_matrix.py --out Data/generated/exp1     # different output dir
"""
import argparse, csv, glob, json, os, subprocess, sys, time

SEEDS     = [11, 12, 13]
FRACTIONS = [0.00, 0.10, 0.25, 0.50]
CONDITION = "hidden_spec_truncation_v1"

SUMMARY_COLS = ["run_id", "seed", "perturbation_fraction", "n_turns_perturbed",
                "perturbed_turns", "total_turns", "correct_recovered",
                "recovery_rate", "task_success", "failed_transfers",
                "termination_reason", "seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Data/generated/exp1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    here = os.path.dirname(os.path.abspath(__file__))
    gen  = os.path.join(here, "generate.py")

    runs, t0 = [], time.time()
    for frac in FRACTIONS:
        for seed in SEEDS:
            print(f"\n{'='*70}\n  fraction {frac:.2f}   seed {seed}\n{'='*70}", flush=True)
            r = subprocess.run([sys.executable, gen,
                                "--seed", str(seed),
                                "--fraction", str(frac),
                                "--condition", CONDITION,
                                "--out", a.out],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(r.stdout, end="")
            if r.returncode != 0:
                print(f"  run returned {r.returncode}")
            paths = [ln.split("=", 1)[1].strip() for ln in r.stdout.splitlines()
                     if ln.startswith("MANIFEST=")]
            if not paths:
                print("  no manifest written; skipping in summary")
                continue
            m = json.load(open(paths[-1]))
            e, p = m.get("ended", {}), m.get("perturbation", {})
            runs.append({
                "run_id": m["conv_id"], "seed": m["sampling"]["seed"],
                "perturbation_fraction": p.get("fraction"),
                "n_turns_perturbed": p.get("n_turns_perturbed"),
                "perturbed_turns": " ".join(str(t) for t in p.get("turns", [])),
                "total_turns": e.get("turns_completed"),
                "correct_recovered": e.get("recovered_count"),
                "recovery_rate": round((e.get("recovered_count") or 0) / m["task"]["n_facts"], 4),
                "task_success": e.get("task_success"),
                "failed_transfers": e.get("failed_transfers"),
                "termination_reason": e.get("reason"),
                "seconds": e.get("seconds"),
            })

    path = os.path.join(a.out, "summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(runs)

    print(f"\n{len(runs)} runs in {time.time()-t0:.0f}s -> {path}\n")
    for r in runs:
        print(f"  frac {r['perturbation_fraction']:.2f}  seed {r['seed']}  "
              f"turns {r['total_turns']:3}  recovered {r['correct_recovered']:2}/24  "
              f"failed {r['failed_transfers']:3}  {r['termination_reason']}")


if __name__ == "__main__":
    main()
