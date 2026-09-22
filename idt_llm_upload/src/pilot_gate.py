"""Pilot gate diagnostics for one generated run.

Uses the project's own tokenizer and entropy implementation (src/msg_entropy.py),
not a substitute. Reports the numbers that decide whether a task can carry the
Omega measurement.

    python src/pilot_gate.py --run Data/generated/inc1/<run>.csv --tokenizer <path>
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from msg_entropy import Tokenizer, message_stats     # the project's implementation

# reference values from the 43-conversation natural corpus, Llama-2 tokens
NATURAL = {"H_prompt_sd": 1.31, "H_response_sd": 0.80,
           "Om_model_sd": 1.49, "Om_user_sd": 0.77,
           "prompt_tokens": 41, "response_tokens": 363}


def sd(x):
    if len(x) < 2:
        return float("nan")
    m = sum(x) / len(x)
    return (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    a = ap.parse_args()

    d = pd.read_csv(a.run).sort_values("Turn")
    tk = Tokenizer(a.tokenizer)

    P = [message_stats(tk.ids(t)) for t in d.Prompt]
    R = [message_stats(tk.ids(t)) for t in d.Response]
    Hp, Hr = [m["H"] for m in P], [m["H"] for m in R]
    Np, Nr = [m["tokens"] for m in P], [m["tokens"] for m in R]

    om_m = [Hp[i + 1] - Hp[i] for i in range(len(Hp) - 1)]
    om_u = [Hr[i + 1] - Hr[i] for i in range(len(Hr) - 1)]
    z_m = sum(1 for v in om_m if abs(v) < 1e-12) / max(len(om_m), 1)
    z_u = sum(1 for v in om_u if abs(v) < 1e-12) / max(len(om_u), 1)

    print(f"run: {os.path.basename(a.run)}")
    print(f"tokenizer: {tk.name}\n")
    print(f"  turns                     {len(d)}")
    if "cum_recovered" in d and d.cum_recovered.notna().any():
        print(f"  elements recovered        {int(d.cum_recovered.iloc[-1])}")
        print(f"  final effectiveness       {d.effectiveness_Et.iloc[-1]}")
    print()
    print(f"  {'':26} {'this run':>10} {'natural':>10} {'gate':>10}")
    rows = [("median prompt tokens",  sorted(Np)[len(Np)//2], NATURAL["prompt_tokens"], ""),
            ("median response tokens", sorted(Nr)[len(Nr)//2], NATURAL["response_tokens"], ">= 20"),
            ("H prompts sd",   sd(Hp), NATURAL["H_prompt_sd"], ""),
            ("H responses sd", sd(Hr), NATURAL["H_response_sd"], ""),
            ("Om_model sd",    sd(om_m), NATURAL["Om_model_sd"], ">= 0.75"),
            ("Om_user sd",     sd(om_u), NATURAL["Om_user_sd"], ""),
            ("Om_model zero fraction", z_m, 0.0, "< 0.10"),
            ("Om_user zero fraction",  z_u, 0.0, "< 0.10")]
    for name, v, nat, gate in rows:
        print(f"  {name:26} {v:10.3f} {nat:10.2f} {gate:>10}")

    print("\n  gate:")
    checks = [("median response tokens >= 20", sorted(Nr)[len(Nr)//2] >= 20),
              ("Om_model sd >= 0.75 (half of natural)", sd(om_m) >= 0.75),
              ("Om_model zeros < 10%", z_m < 0.10),
              ("Om_user zeros < 10%", z_u < 0.10)]
    for label, ok in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n  {'task can carry the measurement' if all(o for _, o in checks) else 'task does not clear the gate'}")


if __name__ == "__main__":
    main()
