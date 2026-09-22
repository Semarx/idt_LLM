"""
omega_advancement_test_v1  -- generation

Preregistered test of one hypothesis:

    |dOmega| is smaller on turns where the conversation recovers new
    ground-truth information than on turns where it stalls.

Nothing is varied. The decoder is fixed, the instruction is fixed, there is no
controller. Probes 1-4 established that neither the decoder nor the length
instruction moves Omega outside its do-nothing band, so both are held constant
here rather than treated as factors.

This module GENERATES only. It computes no entropy, no Omega and no statistic.
Analysis lives in omega_advancement_analyze.py and is run separately, so that
the null validation can be recorded before the real result is opened.

Harness rule that replaces the earlier stalling behaviour
--------------------------------------------------------
Before every asking-agent turn the harness prepends exactly one line:

    Current event to investigate: T-XX

The event is chosen deterministically by round-robin over the events that still
have at least one unrecovered field. The cursor advances every turn regardless
of outcome, so a hard event is revisited once per cycle rather than hammered.
No missing value is ever named, and no field is ever identified as incomplete.

Turn 1 is the fixed opening, identical in all eight conversations, so that
H_prompt_1 is a common anchor. Round-robin steering begins at turn 2 with T-01.
Turn 1 is excluded from all Omega analysis in any case.

Frozen configuration
--------------------
    seeds            11 12 13 14 15 16 17 18
    max turns        25, early stop only on all 60 elements recovered
    temperature      0.7   |  top_p 0.9  |  top_k 40     both models
    num_predict      1200                                both models
    num_ctx          auto (smallest native of the pair)
    history          full, no windowing
    truncation       strict guard, aborts rather than silently dropping
    per-turn seed    seed + turn
"""
import argparse, csv, json, os, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate as G
import task_hidden_incident as INC

EXPERIMENT_ID = "omega_advancement_test_v1"

SEEDS       = [11, 12, 13, 14, 15, 16, 17, 18]
MAX_TURNS   = 120
TEMPERATURE = 0.7
TOP_P       = 0.9
TOP_K       = 40
NUM_PREDICT = 1200

COLUMNS = [
    "experiment_id", "run_id", "seed", "turn", "event_id",
    "prompt", "response", "answer_delivered",
    "prompt_tokens_llama2", "response_tokens_llama2",     # filled by the analyser
    "prompt_chars", "response_chars",
    "newly_recovered", "newly_recovered_count", "cumulative_recovered_count",
    "advance", "effectiveness_Et",
    # generation-side bookkeeping
    "ask_eval_count", "ans_eval_count",
    "ask_ceiling_hit", "ans_ceiling_hit", "ceiling_hit",
    "ask_prompt_eval_count", "ans_prompt_eval_count",
    "ctx_limit", "ctx_headroom_tokens", "context_flag",
    # compliance diagnostic (item B) - never used in the primary test
    "n_question_marks", "single_question_compliant",
]

# The asking role differs from the baseline task only in the questioning rule:
# one question, about the named event. Everything else is the task module's.
ASKING_ROLE = """You are an investigator reconstructing a hidden technical incident.

Another agent has the complete chronological incident record. Your task is to reconstruct the incident as accurately as possible by asking questions.

Before each of your turns you will be told which event to investigate, in the form "Current event to investigate: T-XX".

Rules:

- Ask exactly one focused question, about the named event only.
- Ask about that event's actor, action, target, or consequence.
- Use what you have already learned to decide how to phrase the question.
- Do not ask about other events.
- Do not ask for the entire incident.
- Do not invent facts."""

ANSWERING_ROLE = INC.ANSWERING_ROLE          # unchanged from the frozen task
OPENING        = INC.OPENING                 # unchanged fixed opening

STEER = "Current event to investigate: {event}"


# ---------------------------------------------------------------------------
# round-robin over unresolved events - deterministic, seed-independent
# ---------------------------------------------------------------------------

class RoundRobin:
    def __init__(self):
        self.cursor = 0

    def unresolved(self, recovered):
        return [e for e in INC.EVENT_IDS
                if any((e, f) not in recovered for f in INC.FIELDS)]

    def next_event(self, recovered):
        """Next unresolved event in fixed order. The cursor advances every
        call, resolved or not, so hard events are revisited once per cycle."""
        u = set(self.unresolved(recovered))
        if not u:
            return None
        for _ in range(len(INC.EVENT_IDS)):
            e = INC.EVENT_IDS[self.cursor]
            self.cursor = (self.cursor + 1) % len(INC.EVENT_IDS)
            if e in u:
                return e
        return None


# ---------------------------------------------------------------------------

def chat_fixed(model, messages, seed):
    """One Ollama call at the frozen decoder settings. Uses generate.chat so
    the request path is identical to the rest of the project."""
    keep = (G.TEMPERATURE, G.TOP_P, G.TOP_K, G.NUM_PREDICT)
    G.TEMPERATURE, G.TOP_P, G.TOP_K, G.NUM_PREDICT = (
        TEMPERATURE, TOP_P, TOP_K, NUM_PREDICT)
    try:
        return G.chat(model, messages, seed)
    finally:
        G.TEMPERATURE, G.TOP_P, G.TOP_K, G.NUM_PREDICT = keep


def run_conversation(seed, tag, out_dir):
    run_id = f"adv_s{seed}_{tag}"
    print(f"\n{'='*62}\n=== seed {seed}   run {run_id}\n{'='*62}")

    rr = RoundRobin()
    asking_msgs    = [{"role": "system", "content": ASKING_ROLE}]
    answering_msgs = [{"role": "system", "content": ANSWERING_ROLE}]

    recovered = set()
    rows = []

    _per_path = os.path.join(out_dir, f"{EXPERIMENT_ID}_s{seed}.csv")

    # ---- resume: replay a partial file, if one is there ----------------
    saved = []
    if os.path.exists(_per_path):
        with open(_per_path, newline="", encoding="utf-8") as _rf:
            for r in csv.DictReader(_rf):
                # drop a torn final line
                if not r.get("turn") or not str(r.get("response", "")).strip():
                    break
                try:
                    r["turn"] = int(r["turn"])
                except (TypeError, ValueError):
                    break
                if r["turn"] != len(saved) + 1:      # must be contiguous
                    break
                saved.append(r)

    start_turn = len(saved) + 1
    if saved:
        print(f"  resuming from turn {start_turn} "
              f"({len(saved)} turns replayed from {os.path.basename(_per_path)})")
        for i, r in enumerate(saved):
            # replayed rows come back from CSV as strings; the summary block
            # in main() sums these, so coerce them to match fresh rows
            for _c in ("newly_recovered_count", "cumulative_recovered_count",
                       "advance", "ask_ceiling_hit", "ans_ceiling_hit",
                       "ceiling_hit", "context_flag", "n_question_marks",
                       "single_question_compliant", "prompt_chars",
                       "response_chars", "ctx_limit", "ctx_headroom_tokens"):
                r[_c] = int(r[_c]) if str(r.get(_c, "")).strip() != "" else 0
            r["effectiveness_Et"] = float(r["effectiveness_Et"])
            rows.append(r)
            answering_msgs.append({"role": "user", "content": r["prompt"]})
            answering_msgs.append({"role": "assistant", "content": r["response"]})
            recovered |= INC.elements_in(r["response"])
            if i + 1 < len(saved):
                ev = rr.next_event(recovered)          # advances the cursor
                asking_msgs.append({"role": "user",
                                    "content": r["response"] + "\n\n"
                                               + STEER.format(event=ev)})
                asking_msgs.append({"role": "assistant",
                                    "content": saved[i + 1]["prompt"]})
        print(f"  replayed state: {len(recovered)}/{INC.N_ELEMENTS} elements, "
              f"round-robin cursor {rr.cursor}")

    _pf = open(_per_path, "a" if saved else "w", newline="", encoding="utf-8")
    _pw = csv.DictWriter(_pf, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
    if not saved:
        _pw.writeheader(); _pf.flush(); os.fsync(_pf.fileno())

    prompt = OPENING
    event_id = ""                    # turn 1 is the fixed opening
    end_reason = "max_turns"
    headroom = int(G.CTX_HEADROOM * G.CTX)

    # turn t's prompt is generated at the end of turn t-1, so the asking-side
    # counts arrive one turn early and are carried forward. Turn 1's prompt is
    # the fixed opening and has none.
    pending_ask = {"ask_eval_count": "", "ask_ceiling_hit": 0,
                   "ask_prompt_eval_count": ""}

    t0 = time.time()

    if saved:
        # the prompt that opens the resume turn is regenerated from the
        # replayed history at its own per-turn seed, exactly as it would
        # have been at the end of the last completed turn
        event_id = rr.next_event(recovered)
        if event_id is None:
            _pf.close()
            with open(os.path.join(out_dir, f"{EXPERIMENT_ID}_s{seed}.done"), "w") as _d:
                _d.write(f"no_unresolved_events {len(rows)}\n")
            return rows, "no_unresolved_events", time.time() - t0
        asking_msgs.append({"role": "user",
                            "content": saved[-1]["response"] + "\n\n"
                                       + STEER.format(event=event_id)})
        prompt, ask_read, ask_eval = chat_fixed(
            G.ASKING_MODEL, asking_msgs, seed + start_turn - 1)
        G.guard("asking", G.ASKING_MODEL, ask_read, start_turn - 1)
        asking_msgs.append({"role": "assistant", "content": prompt})
        pending_ask = {"ask_eval_count": ask_eval,
                       "ask_ceiling_hit": int(ask_eval >= NUM_PREDICT),
                       "ask_prompt_eval_count": ask_read}

    for turn in range(start_turn, MAX_TURNS + 1):
        # --- answering agent ------------------------------------------------
        answering_msgs.append({"role": "user", "content": prompt})
        response, ans_read, ans_eval = chat_fixed(
            G.ANSWERING_MODEL, answering_msgs, seed + turn)
        G.guard("answering", G.ANSWERING_MODEL, ans_read, turn)
        answering_msgs.append({"role": "assistant", "content": response})

        # no perturbation in this experiment; delivered == generated
        answer_delivered = response

        # --- deterministic scoring on the delivered text --------------------
        before = set(recovered)
        recovered |= INC.elements_in(answer_delivered)
        new = sorted(f"{e}:{f}" for (e, f) in (recovered - before))
        advance = int(len(new) > 0)

        qm = prompt.count("?")
        ceiling = int(ans_eval >= NUM_PREDICT) | int(pending_ask["ask_ceiling_hit"])

        rows.append({
            "experiment_id": EXPERIMENT_ID, "run_id": run_id, "seed": seed,
            "turn": turn, "event_id": event_id,
            "prompt": prompt, "response": response,
            "answer_delivered": answer_delivered,
            "prompt_tokens_llama2": "", "response_tokens_llama2": "",
            "prompt_chars": len(prompt), "response_chars": len(response),
            "newly_recovered": " | ".join(new),
            "newly_recovered_count": len(new),
            "cumulative_recovered_count": len(recovered),
            "advance": advance,
            "effectiveness_Et": round(len(recovered) / INC.N_ELEMENTS, 4),
            "ask_eval_count": pending_ask["ask_eval_count"],
            "ans_eval_count": ans_eval,
            "ask_ceiling_hit": pending_ask["ask_ceiling_hit"],
            "ans_ceiling_hit": int(ans_eval >= NUM_PREDICT),
            "ceiling_hit": ceiling,
            "ask_prompt_eval_count": pending_ask["ask_prompt_eval_count"],
            "ans_prompt_eval_count": ans_read,
            "ctx_limit": G.CTX, "ctx_headroom_tokens": headroom,
            "context_flag": int(ans_read > headroom or
                                (pending_ask["ask_prompt_eval_count"] != ""
                                 and pending_ask["ask_prompt_eval_count"] > headroom)),
            "n_question_marks": qm,
            "single_question_compliant": int(qm == 1),
        })

        _pw.writerow(rows[-1]); _pf.flush(); os.fsync(_pf.fileno())

        print(f"  [{turn:2d}/{MAX_TURNS}] {event_id or 'opening':7s}  "
              f"+{len(new):2d} new  {len(recovered):2d}/{INC.N_ELEMENTS}  "
              f"E={len(recovered)/INC.N_ELEMENTS:.3f}  "
              f"adv={advance}  ans_tok={ans_eval}"
              + ("  CEILING" if ans_eval >= NUM_PREDICT else ""))

        if len(recovered) >= INC.N_ELEMENTS:
            end_reason = "all_elements_recovered"
            print(f"  all {INC.N_ELEMENTS} elements recovered at turn {turn}")
            break
        if turn == MAX_TURNS:
            break

        # --- asking agent, steered by the harness ---------------------------
        event_id = rr.next_event(recovered)
        if event_id is None:
            end_reason = "no_unresolved_events"
            break
        asking_msgs.append({"role": "user",
                            "content": answer_delivered + "\n\n"
                                       + STEER.format(event=event_id)})
        prompt, ask_read, ask_eval = chat_fixed(
            G.ASKING_MODEL, asking_msgs, seed + turn)
        G.guard("asking", G.ASKING_MODEL, ask_read, turn)
        asking_msgs.append({"role": "assistant", "content": prompt})

        # these counts describe the prompt that opens the NEXT turn
        pending_ask = {"ask_eval_count": ask_eval,
                       "ask_ceiling_hit": int(ask_eval >= NUM_PREDICT),
                       "ask_prompt_eval_count": ask_read}

    _pf.close()
    with open(os.path.join(out_dir, f"{EXPERIMENT_ID}_s{seed}.done"), "w") as _d:
        _d.write(f"{end_reason} {len(rows)}\n")
    return rows, end_reason, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Data/advancement")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    a = ap.parse_args()

    if G.EXPERIMENT != "hidden_incident_baseline_v1":
        sys.exit(f"generate.EXPERIMENT is '{G.EXPERIMENT}'; this experiment "
                 f"expects the incident task to be active.")

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    os.makedirs(a.out, exist_ok=True)
    specs = G.check_server()
    tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "|dOmega| is smaller on advancing turns than on stalled turns",
        "preregistered_statistic": "D = mean(|dOmega| | advance=1) - mean(|dOmega| | advance=0)",
        "preregistered_prediction": "D < 0",
        "omega_definition": {
            "Omega_prompt_t":   "H_prompt_t - H_prompt_(t-1)",
            "Omega_response_t": "H_response_t - H_response_(t-1)",
            "dOmega_t":         "Omega_response_t - Omega_prompt_t",
            "turn_1":           "excluded from all Omega analysis",
        },
        "asking_model": G.ASKING_MODEL, "answering_model": G.ANSWERING_MODEL,
        "model_specs": specs,
        "decoder": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
                    "num_predict": NUM_PREDICT, "both_models": True},
        "num_ctx_resolved": G.CTX,
        "ctx_headroom_tokens": int(G.CTX_HEADROOM * G.CTX),
        "strict_no_truncation": G.STRICT_NO_TRUNCATION,
        "seeds": seeds, "max_turns": MAX_TURNS,
        "early_stop": "only when all 60 elements are recovered",
        "task": {"task_id": INC.TASK_ID, "name": INC.INCIDENT_NAME,
                 "n_events": len(INC.INCIDENT), "n_elements": INC.N_ELEMENTS,
                 "fields": INC.FIELDS, "events": INC.INCIDENT},
        "steering": {"rule": "round-robin over unresolved events, cursor "
                             "advances every turn",
                     "message": STEER,
                     "begins_at_turn": 2,
                     "turn_1": "fixed opening, identical in all conversations"},
        "asking_role": ASKING_ROLE, "answering_role": ANSWERING_ROLE,
        "frozen_handling_rules": {
            "A_flagged_conversations": "included in the primary pooled D; "
                                       "exclusion reported only as secondary "
                                       "robustness",
            "B_one_question": "instruction only; prompts are never edited, "
                              "since editing would change the text whose H is "
                              "measured. n_question_marks logged as diagnostic",
            "C_turn_index": "advance rate by turn and |dOmega| vs turn index "
                            "logged as diagnostics, outside the primary test",
        },
        "exclusions": ["turn 1", "turns hitting num_predict and the turn after",
                       "turns flagged by the context guard"],
    }
    with open(os.path.join(a.out, f"{EXPERIMENT_ID}_manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)

    combined = os.path.join(a.out, f"{EXPERIMENT_ID}_turns.csv")
    cfh = open(combined, "w", newline="", encoding="utf-8")
    cw = csv.DictWriter(cfh, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
    cw.writeheader()

    summary_rows = []
    t_all = time.time()
    for seed in seeds:
        per  = os.path.join(a.out, f"{EXPERIMENT_ID}_s{seed}.csv")
        done = os.path.join(a.out, f"{EXPERIMENT_ID}_s{seed}.done")

        if os.path.exists(done) and os.path.exists(per):
            with open(per, newline="", encoding="utf-8") as pf:
                rows = list(csv.DictReader(pf))
            for r in rows:
                for c in ("turn", "newly_recovered_count",
                          "cumulative_recovered_count", "advance",
                          "ceiling_hit", "context_flag",
                          "single_question_compliant"):
                    r[c] = int(r[c]) if r[c] not in ("", None) else 0
                r["effectiveness_Et"] = float(r["effectiveness_Et"])
            end_reason = open(done).read().split()[0]
            wall = 0.0
            print(f"\n=== seed {seed}: already complete ({len(rows)} turns), skipping")
        else:
            rows, end_reason, wall = run_conversation(seed, tag, a.out)

        # the per-seed file is written turn by turn inside run_conversation
        for r in rows:
            cw.writerow(r)
        cfh.flush()

        adv = sum(r["advance"] for r in rows)
        summary_rows.append({
            "experiment_id": EXPERIMENT_ID, "run_id": rows[0]["run_id"],
            "seed": seed, "turns": len(rows), "end_reason": end_reason,
            "advancing_turns": adv, "stalled_turns": len(rows) - adv,
            "elements_recovered": rows[-1]["cumulative_recovered_count"],
            "effectiveness_final": rows[-1]["effectiveness_Et"],
            "ceiling_hits": sum(r["ceiling_hit"] for r in rows),
            "context_flags": sum(r["context_flag"] for r in rows),
            "single_question_rate": round(
                sum(r["single_question_compliant"] for r in rows) / len(rows), 3),
            "wall_seconds": round(wall, 1),
        })
        print(f"  -> {end_reason}: {adv} advancing / {len(rows)-adv} stalled, "
              f"E={rows[-1]['effectiveness_Et']:.3f}  ({wall/60:.1f} min)")

    cfh.close()
    spath = os.path.join(a.out, f"{EXPERIMENT_ID}_runs.csv")
    with open(spath, "w", newline="", encoding="utf-8") as sf:
        sw = csv.DictWriter(sf, fieldnames=list(summary_rows[0]),
                            quoting=csv.QUOTE_ALL)
        sw.writeheader()
        for r in summary_rows:
            sw.writerow(r)

    tot_adv = sum(r["advancing_turns"] for r in summary_rows)
    tot_turn = sum(r["turns"] for r in summary_rows)
    print(f"\n{'='*62}")
    print(f"done in {(time.time()-t_all)/60:.1f} min")
    print(f"  turns {tot_turn}   advancing {tot_adv}   "
          f"stalled {tot_turn - tot_adv}")
    print(f"  {combined}")
    print(f"  {spath}")
    print(f"\nnext:  python src/omega_advancement_analyze.py "
          f"--turns {combined}")


if __name__ == "__main__":
    main()
