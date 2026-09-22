"""
Generate a controlled two-model conversation and log it turn by turn.

Two local models talk to each other through Ollama. One plays the asking side
(its messages become Prompt), the other the answering side (Response). Every turn
is appended to the CSV as it happens, so nothing is lost if a run is interrupted
and there is no assembly step at the end.

Output matches the schema of turns_all_ts.csv, so build_streams.py, screens.py and
omega.py read generated conversations with no changes.

Run:  python src/generate.py
Everything you need to change is in the CONFIG block below.
"""

# ============================================================================
#  CONFIG  — the only part you edit
# ============================================================================

# EXPERIMENT MODE
#   None                    -> free conversation, using the OPENING/ROLE text below
#   "hidden_spec_baseline"  -> the Orion-24 recovery task (src/task_hidden_spec.py)
#                              supplies the spec, both system prompts and the opening,
#                              and the run stops early on TASK COMPLETE.
#   "hidden_spec_truncation_v1"      -> same task, with a controlled seam truncation
#   "hidden_incident_baseline_v1"    -> incident reconstruction, 60 ground-truth
#                                       elements (src/task_hidden_incident.py)
EXPERIMENT  = "hidden_incident_baseline_v1"

CONDITION   = "intact"                     # name of this run; becomes the conversation title
N_TURNS     = 60                           # a turn is one prompt + one response

# --- perturbation: seam truncation --------------------------------------
# Fraction of the 24 task turns whose answer is truncated before it reaches the
# asking agent. The answering agent still generates its full answer; only what
# crosses the line is altered.
PERTURBATION_TYPE     = "truncation"
PERTURBATION_FRACTION = 0.00           # 0.00 | 0.10 | 0.25 | 0.50
PERTURBATION_SEED     = None           # None -> derived from SEED

# Pin exact versions. `latest` is a moving tag - replace it with the explicit tag
# once `ollama show llama3.1:latest` tells you what it resolves to. The digest of
# whatever is actually used is recorded in the manifest either way.
ASKING_MODEL    = "mistral:7b"        # its messages become Prompt
ANSWERING_MODEL = "llama3.1:latest"   # its messages become Response

OPENING = (
    "I want to understand how memory works in the brain. "
    "Start with the difference between short-term and long-term memory."
)

ASKING_ROLE = (
    "You are a curious researcher in an ongoing conversation. "
    "Ask one substantive follow-up question at a time, building on what you were just told. "
    "Stay on the subject and keep going deeper. "
    "Never thank, never summarise, never close the conversation. "
    "Reply with the question only."
)

ANSWERING_ROLE = (
    "You are a knowledgeable assistant. Answer the question directly and substantively. "
    "Do not ask questions back. Do not offer to help further."
)

# turn number -> text that replaces the asking model's message at that turn.
# Leave empty for a baseline run. Example: {31: "Let's talk about something else..."}
INJECTIONS = {}

SEED        = 11
TEMPERATURE = 0.7
TOP_P       = 0.9
TOP_K       = 40       # pinned, not left to the server default
NUM_PREDICT = 300      # max tokens per message

# Context window given to each model.
#   "auto"  -> the largest both models natively support (recommended for a baseline)
#   <int>   -> that value
NUM_CTX     = "auto"

# A baseline must contain NO hidden context loss: neither model may silently drop
# earlier turns. The run aborts the moment either side's history approaches the
# window, rather than truncating quietly. Set False only for a deliberate
# truncation condition.
STRICT_NO_TRUNCATION = True
CTX_HEADROOM         = 0.90   # abort once history exceeds this fraction of the window

import os as _os
OLLAMA  = _os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OUT_DIR = "Data/generated"

CTX = None   # resolved at startup from the models' native windows

# ============================================================================
#  Nothing below here needs editing
# ============================================================================

import csv, json, os, sys, time, urllib.request, urllib.error
from datetime import datetime, timezone

TASK = None          # ORION-24 spec-recovery family (retained, unchanged)
INC  = None          # incident-reconstruction family
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if EXPERIMENT in ("hidden_spec_baseline", "hidden_spec_truncation_v1"):
    import task_hidden_spec as TASK
    ASKING_ROLE, ANSWERING_ROLE, OPENING = TASK.ASKING_ROLE, TASK.ANSWERING_ROLE, TASK.OPENING
elif EXPERIMENT == "hidden_incident_baseline_v1":
    import task_hidden_incident as INC
    ASKING_ROLE, ANSWERING_ROLE, OPENING = INC.ASKING_ROLE, INC.ANSWERING_ROLE, INC.OPENING
elif EXPERIMENT is not None:
    sys.exit(f"Unknown EXPERIMENT: {EXPERIMENT}")

COLUMNS = ["uid", "platform", "conv_id", "title", "Turn", "Prompt", "Response",
           "timestamp", "prompt_chars", "response_chars",
           "run_id", "task_id", "condition",
           "asking_raw", "answering_raw",
           "ctx_tokens_answering", "ctx_tokens_asking",
           "field_requested", "remaining_fields_before",
           "remaining_fields_after", "recovered_count",
           "answer_delivered", "perturbation_type", "perturbation_fraction",
           "perturbation_seed", "turn_perturbed", "transfer_correct",
           "expected_value",
           # incident-reconstruction task
           "new_elements", "n_new_elements", "cum_recovered", "effectiveness_Et",
           "incomplete_events_before", "incomplete_events_after", "conflicts"]


def chat(model, messages, seed):
    """One call to Ollama. Returns the message text."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"seed": seed, "temperature": TEMPERATURE, "top_p": TOP_P,
                    "top_k": TOP_K, "num_predict": NUM_PREDICT, "num_ctx": CTX},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    # prompt_eval_count is how many tokens the model actually read this call -
    # the exact measure of whether the history still fits.
    return d["message"]["content"].strip(), d.get("prompt_eval_count", 0), d.get("eval_count", 0)


def guard(side, model, used, turn, on_stop=None):
    """Abort rather than let the server silently drop earlier turns."""
    if not STRICT_NO_TRUNCATION:
        return
    if used >= CTX_HEADROOM * CTX:
        sys.exit(
            f"\n  STOPPED at turn {turn}: {side} side ({model}) read {used} tokens, "
            f"which is {used/CTX:.0%} of the {CTX}-token window.\n"
            f"  Continuing would silently drop earlier turns, so the run is not a "
            f"clean baseline.\n"
            f"  Everything up to turn {turn-1} is already saved and usable.\n"
            f"  Options: fewer turns, smaller NUM_PREDICT, a longer-context model, "
            f"or set STRICT_NO_TRUNCATION = False if truncation is the condition "
            f"being tested.")


def model_spec(model):
    """Everything Ollama knows about a model - recorded in the manifest."""
    body = json.dumps({"model": model}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/show", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    det = d.get("details", {})
    info = d.get("model_info", {})
    arch = info.get("general.architecture", det.get("family", ""))
    return {
        "name": model,
        "family": det.get("family"),
        "parameter_size": det.get("parameter_size"),
        "quantization": det.get("quantization_level"),
        "format": det.get("format"),
        "architecture": arch,
        "context_length": info.get(f"{arch}.context_length"),
        "embedding_length": info.get(f"{arch}.embedding_length"),
        "block_count": info.get(f"{arch}.block_count"),
        "vocab_size": info.get(f"{arch}.vocab_size") or info.get("tokenizer.ggml.tokens.length"),
        "parameter_count": info.get("general.parameter_count"),
        "modified": d.get("modified_at"),
        "digest": d.get("digest"),
    }


def check_server():
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
            have = {m["name"] for m in json.loads(r.read())["models"]}
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {OLLAMA}. Is the Ollama app running?  ({e})")
    for m in (ASKING_MODEL, ANSWERING_MODEL):
        if m not in have:
            sys.exit(f"Model '{m}' is not installed. `ollama list` shows: {sorted(have)}")
    specs = {}
    for role, m in (("asking", ASKING_MODEL), ("answering", ANSWERING_MODEL)):
        sp = model_spec(m)
        specs[role] = sp
        print(f"  {role:9s} {m:22s} {sp['parameter_size'] or '?':>6s}  "
              f"{sp['quantization'] or '?':10s} native ctx {sp['context_length'] or '?'}")

    global CTX
    native = [sp["context_length"] for sp in specs.values() if sp["context_length"]]
    if NUM_CTX == "auto":
        if not native:
            sys.exit("Could not read native context length; set NUM_CTX to a number.")
        CTX = min(native)
        print(f"\n  context window: {CTX} tokens (smallest natively supported by the pair)")
    else:
        CTX = int(NUM_CTX)
        if native and CTX > min(native):
            print(f"\n  WARNING: NUM_CTX {CTX} exceeds the smallest native window "
                  f"{min(native)}; the model may not use it all.")
        else:
            print(f"\n  context window: {CTX} tokens (set explicitly)")
    print(f"  strict no-truncation: {STRICT_NO_TRUNCATION}"
          f"  (abort above {int(CTX_HEADROOM*CTX)} tokens read)\n")
    return specs


def apply_overrides():
    """Optional command-line overrides, so a run matrix can drive one script."""
    global SEED, PERTURBATION_FRACTION, PERTURBATION_SEED, CONDITION, OUT_DIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--fraction", type=float)
    ap.add_argument("--perturbation-seed", type=int)
    ap.add_argument("--condition")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.seed is not None:              SEED = a.seed
    if a.fraction is not None:          PERTURBATION_FRACTION = a.fraction
    if a.perturbation_seed is not None: PERTURBATION_SEED = a.perturbation_seed
    if a.condition:                     CONDITION = a.condition
    if a.out:                           OUT_DIR = a.out


def select_perturbed_turns(n_fields, fraction, pseed):
    """Deterministic, before the dialogue begins, from a dedicated RNG.

    Eligible turns are 1..n_fields - one per required field. round(n*fraction)
    turns are drawn without replacement. Never adaptive: no generated text and no
    task performance enters this choice.
    """
    import random
    k = int(round(n_fields * fraction))
    if k <= 0:
        return []
    rng = random.Random(pseed)
    return sorted(rng.sample(range(1, n_fields + 1), k))


def main():
    apply_overrides()
    specs = check_server()
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    conv_id = f"gen_{CONDITION}_f{int(round(PERTURBATION_FRACTION*100)):03d}_s{SEED}_{stamp}"
    csv_path = os.path.join(OUT_DIR, f"{conv_id}.csv")

    # external checklist - the harness owns it, not the asking agent
    remaining = list(TASK.REQUIRED_FIELDS) if TASK else []
    recovered_count = 0
    failed_transfers = 0

    # incident-reconstruction state, owned by the harness
    recovered_elems = set()
    all_conflicts = []

    # perturbation plan - fixed before any generation
    pseed = PERTURBATION_SEED if PERTURBATION_SEED is not None else SEED
    perturbed_turns = (select_perturbed_turns(len(TASK.REQUIRED_FIELDS),
                                              PERTURBATION_FRACTION, pseed)
                       if (TASK and EXPERIMENT == "hidden_spec_truncation_v1") else [])
    if TASK:
        print(f"  perturbation: {PERTURBATION_TYPE}  fraction {PERTURBATION_FRACTION:.2f}  "
              f"seed {pseed}  turns {perturbed_turns}\n")

    manifest = {
        "conv_id": conv_id, "condition": CONDITION, "n_turns": N_TURNS,
        "asking_model": ASKING_MODEL, "answering_model": ANSWERING_MODEL,
        "model_specs": specs,
        "sampling": {"seed": SEED, "temperature": TEMPERATURE, "top_p": TOP_P,
                     "top_k": TOP_K, "num_predict": NUM_PREDICT,
                     "num_ctx_setting": NUM_CTX, "num_ctx_resolved": CTX,
                     "seed_per_turn": "SEED + turn"},
        "strict_no_truncation": STRICT_NO_TRUNCATION,
        "ctx_headroom": CTX_HEADROOM,
        "experiment": EXPERIMENT,
        "perturbation": {"type": PERTURBATION_TYPE,
                         "fraction": PERTURBATION_FRACTION,
                         "seed": pseed,
                         "unit_rule": (TASK.TRUNCATION_UNIT if TASK else None),
                         "rule": "deliver ceil(n_units/2) leading units",
                         "turns": perturbed_turns,
                         "n_turns_perturbed": len(perturbed_turns)},
        "task": ({"task_id": TASK.TASK_ID, "spec_name": TASK.SPEC_NAME,
                  "n_facts": len(TASK.SPEC_FACTS),
                  "facts": [{"property": k, "value": v} for k, v in TASK.SPEC_FACTS],
                  "completion_token": TASK.COMPLETION_TOKEN,
                  "not_specified_token": TASK.NOT_SPECIFIED} if TASK else None),
        "incident": ({"task_id": INC.TASK_ID, "name": INC.INCIDENT_NAME,
                      "n_events": len(INC.INCIDENT), "n_elements": INC.N_ELEMENTS,
                      "fields": INC.FIELDS, "events": INC.INCIDENT,
                      "ground_truth": [{"event_id": k[0], "field": k[1], "value": v}
                                       for k, v in INC.GROUND_TRUTH.items()]}
                     if INC else None),
        "ollama_endpoint": OLLAMA,
        "python": sys.version.split()[0],
        "opening": OPENING, "asking_role": ASKING_ROLE,
        "answering_role": ANSWERING_ROLE, "injections": INJECTIONS,
        "seed": SEED, "temperature": TEMPERATURE, "top_p": TOP_P,
        "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(OUT_DIR, f"{conv_id}.manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    f = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
    w.writeheader()

    # each side keeps its own view of the conversation
    asking_msgs   = [{"role": "system", "content": ASKING_ROLE}]
    answering_msgs = [{"role": "system", "content": ANSWERING_ROLE}]


    prompt = OPENING
    prev_ask_read = 0          # asking-side context read that produced this prompt
    end_reason = "max_turns"
    turns_done = 0
    t0 = time.time()

    try:
      for turn in range(1, N_TURNS + 1):
          if turn in INJECTIONS:
              prompt = INJECTIONS[turn]
              asking_msgs.append({"role": "assistant", "content": prompt})

          remaining_before = list(remaining)
          field_requested = TASK.match_field(prompt, remaining) if TASK else None

          answering_msgs.append({"role": "user", "content": prompt})
          response, ans_read, _ = chat(ANSWERING_MODEL, answering_msgs, SEED + turn)
          guard("answering", ANSWERING_MODEL, ans_read, turn)
          # the answering model's own history holds what it actually generated
          answering_msgs.append({"role": "assistant", "content": response})

          # what crosses the line may differ from what was generated
          turn_perturbed = turn in perturbed_turns
          answer_delivered = TASK.truncate_half(response) if (turn_perturbed and TASK) else response

          # scoring is against what the asking agent received, never the raw answer
          expected_value = TASK.EXPECTED.get(field_requested, "") if (TASK and field_requested) else ""
          transfer_correct = TASK.value_present(answer_delivered, field_requested) if TASK else False

          # --- incident task: deterministic element scoring on delivered text ---
          inc_before = INC.incomplete_events(recovered_elems) if INC else []
          new_elems, turn_conflicts = [], []
          if INC:
              found = INC.elements_in(answer_delivered)
              new = sorted(found - recovered_elems)
              recovered_elems |= found
              new_elems = [f"{e}:{f}" for e, f in new]
              for e in INC.EVENT_IDS:
                  for fl in INC.FIELDS:
                      if (e, fl) in found:
                          turn_conflicts += INC.conflicts_in(answer_delivered, e, fl)
              all_conflicts += turn_conflicts
          inc_after = INC.incomplete_events(recovered_elems) if INC else []

          if transfer_correct:
              remaining.remove(field_requested)
              recovered_count += 1
          elif field_requested is not None:
              failed_transfers += 1        # field stays in remaining; it can be asked again

          w.writerow({"uid": f"{conv_id}_{turn:04d}", "platform": "generated",
                      "conv_id": conv_id, "title": CONDITION, "Turn": turn,
                      "Prompt": prompt, "Response": answer_delivered,
                      "timestamp": datetime.now(timezone.utc).isoformat(),
                      "prompt_chars": len(prompt), "response_chars": len(answer_delivered),
                      "run_id": conv_id, "task_id": (TASK.TASK_ID if TASK else ""),
                      "condition": CONDITION,
                      "asking_raw": prompt, "answering_raw": response,
                      "answer_delivered": answer_delivered,
                      "perturbation_type": PERTURBATION_TYPE,
                      "perturbation_fraction": PERTURBATION_FRACTION,
                      "perturbation_seed": pseed,
                      "turn_perturbed": turn_perturbed,
                      "transfer_correct": transfer_correct,
                      "expected_value": expected_value,
                      "new_elements": " | ".join(new_elems),
                      "n_new_elements": len(new_elems),
                      "cum_recovered": len(recovered_elems),
                      "effectiveness_Et": (round(len(recovered_elems)/INC.N_ELEMENTS, 4)
                                           if INC else ""),
                      "incomplete_events_before": " ".join(inc_before),
                      "incomplete_events_after": " ".join(inc_after),
                      "conflicts": " | ".join(sorted(set(turn_conflicts))),
                      "ctx_tokens_answering": ans_read, "ctx_tokens_asking": prev_ask_read,
                      "field_requested": field_requested or "",
                      "remaining_fields_before": " | ".join(remaining_before),
                      "remaining_fields_after": " | ".join(remaining),
                      "recovered_count": recovered_count})
          f.flush()

          print(f"[{turn:3d}/{N_TURNS}]  prompt {len(prompt):5d} ch   "
                f"response {len(response):5d} ch   ctx {ans_read:6d}/{CTX} "
                f"({ans_read/CTX:4.0%})   recovered {recovered_count:2d}/"
                f"{len(TASK.REQUIRED_FIELDS) if TASK else 0}   {time.time()-t0:5.0f}s")
          if INC:
              print(f"      +{len(new_elems):2d} new   {len(recovered_elems):2d}/{INC.N_ELEMENTS} "
                    f"(E={len(recovered_elems)/INC.N_ELEMENTS:.2f})   "
                    f"incomplete: {len(inc_after)} events")
          if TASK:
              mark = "TRUNCATED" if turn_perturbed else ""
              ok = "ok " if transfer_correct else "MISS"
              print(f"      field: {field_requested or '-- no match --':32s} {ok} {mark}")
          print(f"      Q: {prompt[:110]}")
          print(f"      A: {response[:110]}\n")

          turns_done = turn

          if INC is not None and len(recovered_elems) >= INC.N_ELEMENTS:
              end_reason = "all_elements_recovered"
              print(f"\n  all {INC.N_ELEMENTS} elements recovered at turn {turn}")
              break

          if TASK is not None and not remaining:
              end_reason = "all_fields_recovered"
              print(f"\n  all {recovered_count} fields recovered at turn {turn}")
              break

          if TASK is not None and TASK.is_complete(prompt):
              end_reason = "task_complete"
              print(f"  asking agent signalled {TASK.COMPLETION_TOKEN} at turn {turn}")
              break

          if turn == N_TURNS:
              end_reason = "max_turns"
              break

          # the asking model sees the answer and produces the next question
          nxt = answer_delivered
          if TASK is not None:
              nxt = answer_delivered + "\n\n" + TASK.remaining_instruction(remaining)
          elif INC is not None:
              nxt = answer_delivered + "\n\n" + INC.progress_message(recovered_elems)
          asking_msgs.append({"role": "user", "content": nxt})
          prompt, prev_ask_read, _ = chat(ASKING_MODEL, asking_msgs, SEED + turn)
          guard("asking", ASKING_MODEL, prev_ask_read, turn)
          asking_msgs.append({"role": "assistant", "content": prompt})
    except SystemExit as e:
        end_reason = "context_guard"
        print(e)

    f.close()

    manifest["ended"] = {
        "reason": end_reason,
        "turns_completed": turns_done,
        "recovered_count": recovered_count,
        "fields_not_recovered": remaining,
        "failed_transfers": failed_transfers,
        "task_success": (int(len(recovered_elems) >= INC.N_ELEMENTS) if INC
                         else int(len(remaining) == 0)),
        "elements_recovered": len(recovered_elems) if INC else None,
        "effectiveness_final": (round(len(recovered_elems)/INC.N_ELEMENTS, 4) if INC else None),
        "elements_unrecovered": ([f"{e}:{f}" for (e, f) in sorted(set(INC.GROUND_TRUTH) - recovered_elems)]
                                 if INC else None),
        "conflicts_observed": sorted(set(all_conflicts)) if INC else None,
        "seconds": round(time.time() - t0, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(OUT_DIR, f"{conv_id}.manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"\nDone. {turns_done} turns, ended by {end_reason}, in {time.time()-t0:.0f}s")
    print(f"  {csv_path}")
    print(f"  {os.path.join(OUT_DIR, conv_id + '.manifest.json')}")
    print(f"MANIFEST={os.path.join(OUT_DIR, conv_id + '.manifest.json')}")


if __name__ == "__main__":
    main()
