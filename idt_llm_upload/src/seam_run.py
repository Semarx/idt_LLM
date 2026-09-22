"""
seam_run.py — T1 runner. Executes one conversation under one named breach condition
and writes a seam log.

Reuses the proven harness primitives: generate.chat (identical request path),
generate.guard (context guard), task_hidden_incident (the task), and the round-robin
steering from omega_advancement_test that gives 100% single-question compliance.

Records only. Computes no entropy and no Omega — analysis is a separate program so the
recording cannot be tuned to the result.

    python src/seam_run.py --condition none      --seed 11 --out Data/seam/T2
    python src/seam_run.py --condition truncate  --fraction 0.4 --overlap 0.0 \
                           --seed 11 --out Data/seam/T3
    python src/seam_run.py --condition compact   --k-fraction 0.4 --persist \
                           --seed 11 --out Data/seam/T3

Frozen parameters (00_Frozen_Decisions.md): K = 32000, W = 50, 60-turn conversations,
n_min = 10.
"""
import argparse, copy, json, os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate as G
import task_hidden_incident as INC
from omega_advancement_test import RoundRobin, ASKING_ROLE, STEER
from msg_entropy import Tokenizer
from seam_log import (SeamRecorder, BREACHES, carried_forward,
                      K_DECLARED, W_DEFAULT, N_MIN)

W = 50                         # window length in pairs (F6, amended 13 Sep)
TURNS = 60                     # conversation length; must exceed W to yield readings
TEMPERATURE, TOP_P, TOP_K = 0.7, 0.9, 40
NUM_PREDICT = 1200

# Which breaches replace the application's own stored history (persistent) and which
# are applied fresh in transit each turn (transport defects). Declared, not inferred.
PERSISTENT_BY_DEFAULT = {"compact"}


def chat_fixed(model, messages, seed):
    keep = (G.TEMPERATURE, G.TOP_P, G.TOP_K, G.NUM_PREDICT)
    G.TEMPERATURE, G.TOP_P, G.TOP_K, G.NUM_PREDICT = (
        TEMPERATURE, TOP_P, TOP_K, NUM_PREDICT)
    try:
        return G.chat(model, messages, seed)
    finally:
        G.TEMPERATURE, G.TOP_P, G.TOP_K, G.NUM_PREDICT = keep


def run(cond, params, seed, out_dir, tok, onset=1, persist=None):
    if persist is None:
        persist = cond in PERSISTENT_BY_DEFAULT
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"seam_{cond}_s{seed}_{stamp}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, run_id + ".jsonl")

    rec = SeamRecorder(path, tok, run_id, cond,
                       {**params, "seed": seed, "onset_turn": onset,
                        "persist": persist, "temperature": TEMPERATURE,
                        "top_p": TOP_P, "top_k": TOP_K,
                        "num_predict": NUM_PREDICT,
                        "asking_model": G.ASKING_MODEL,
                        "answering_model": G.ANSWERING_MODEL})

    rr = RoundRobin()
    asking_msgs = [{"role": "system", "content": ASKING_ROLE}]
    answering_msgs = [{"role": "system", "content": INC.ANSWERING_ROLE}]
    recovered, prompt, event_id = set(), INC.OPENING, ""
    print(f"\n=== {run_id}   breach={cond} {params}  persist={persist}  onset={onset}")

    t0 = time.time()
    for turn in range(1, TURNS + 1):
        answering_msgs.append({"role": "user", "content": prompt})

        # --- the seam: breach applied between assembly and the call -----------
        if turn >= onset and cond != "none":
            breached, meta = BREACHES[cond](answering_msgs, **params)
        else:
            breached, meta = answering_msgs, {"breach": "none"}

        response, ans_read, ans_eval = chat_fixed(
            G.ANSWERING_MODEL, breached, seed + turn)
        G.guard("answering", G.ANSWERING_MODEL, ans_read, turn)

        # Z = the pair as emitted;  S = the carried-forward region as delivered
        rec.record_turn(turn, prompt, response, carried_forward(breached), meta)

        # a persistent breach replaces the application's own history; a transport
        # defect leaves it intact and is re-applied fresh next turn
        if persist and turn >= onset and cond != "none":
            answering_msgs = copy.deepcopy(breached)
        answering_msgs.append({"role": "assistant", "content": response})

        recovered |= INC.elements_in(response)
        print(f"  [{turn:2d}/{TURNS}] {event_id or 'open':6s} "
              f"pairs_delivered={len(carried_forward(breached)):2d} "
              f"tok={ans_eval:4d}  E={len(recovered)/INC.N_ELEMENTS:.3f}"
              + ("  CEILING" if ans_eval >= NUM_PREDICT else ""))

        if turn == TURNS:
            break
        event_id = rr.next_event(recovered)
        if event_id is None:
            break
        asking_msgs.append({"role": "user",
                            "content": response + "\n\n" + STEER.format(event=event_id)})
        prompt, ask_read, _ = chat_fixed(G.ASKING_MODEL, asking_msgs, seed + turn)
        G.guard("asking", G.ASKING_MODEL, ask_read, turn)
        asking_msgs.append({"role": "assistant", "content": prompt})

    rec.close()
    print(f"  -> {path}   ({(time.time()-t0)/60:.1f} min)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=sorted(BREACHES))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="Data/seam")
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--onset", type=int, default=1,
                    help="first turn at which the breach is applied")
    ap.add_argument("--persist", action="store_true", default=None,
                    help="breach replaces stored history (default: only for compact)")
    ap.add_argument("--fraction", type=float)
    ap.add_argument("--k-fraction", type=float, dest="k_fraction")
    ap.add_argument("--index", type=int)
    ap.add_argument("--scope", default=None)
    a = ap.parse_args()

    params = {k: v for k, v in
              (("fraction", a.fraction), ("k_fraction", a.k_fraction),
               ("index", a.index), ("scope", a.scope)) if v is not None}

    G.check_server()
    tok = Tokenizer(a.tokenizer)
    run(a.condition, params, a.seed, a.out, tok, onset=a.onset, persist=a.persist)


if __name__ == "__main__":
    main()
