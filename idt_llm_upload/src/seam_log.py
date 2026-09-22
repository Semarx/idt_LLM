"""
seam_log.py — T1 instrumentation for the seam-integrity experiments.

Records the two sides of the return-path seam, and injects controlled breaches between
context assembly and the model call.

    Z  = the pairs as EMITTED        (prompt, response) at each turn
    S  = the carried-forward region as DELIVERED, read from the message list at the
         moment of the call

Frozen parameters (00_Frozen_Decisions.md):
    K      = 32000      declared vocabulary
    W      = 50         window length, in pairs
    n_min  = 10         estimator eligibility: minimum tokens per pair

This module RECORDS and BREACHES. It computes no entropy and no Omega — analysis is
separate, so the recording cannot be tuned to the result.

Pair distributions are stored sparsely (token id -> count). A pair of ~200 tokens has at
most 200 non-zero entries out of K=32000, so the log stays small.
"""
import copy, json, os
from collections import Counter

K_DECLARED = 32000
W_DEFAULT = 50
N_MIN = 10

SCHEMA_VERSION = "seam_log/1"


# ---------------------------------------------------------------------------
# reading the two sides
# ---------------------------------------------------------------------------

def carried_forward(msgs):
    """The compared region of an assembled message list.

    msgs = [system, p_1, r_1, ..., p_{t-1}, r_{t-1}, p_t]

    Returns the prior pairs only — system material and the current prompt are outside
    the compared region (F2). Returns a list of (prompt, response) tuples.
    """
    body = msgs[1:]                       # drop system
    if body and body[-1]["role"] == "user":
        body = body[:-1]                  # drop the current prompt
    pairs = []
    for i in range(0, len(body) - 1, 2):
        if body[i]["role"] == "user" and body[i + 1]["role"] == "assistant":
            pairs.append((body[i]["content"], body[i + 1]["content"]))
    return pairs


def sparse_counts(text, tok):
    """Token id -> count for one message. Keys are strings for JSON round-tripping."""
    ids = tok.ids(text)
    return {str(k): v for k, v in Counter(ids).items()}, len(ids)


# ---------------------------------------------------------------------------
# breach injectors — applied to the carried-forward region, in transit
# ---------------------------------------------------------------------------

def breach_none(msgs, **kw):
    return msgs, {"breach": "none"}


def _split(msgs):
    """-> (head, pair_msgs, tail) where pair_msgs is the carried-forward region."""
    head = msgs[:1]
    body = msgs[1:]
    tail = []
    if body and body[-1]["role"] == "user":
        tail = [body[-1]]
        body = body[:-1]
    return head, body, tail


def breach_truncate(msgs, fraction=0.4, **kw):
    """Drop the OLDEST fraction of carried-forward pairs. Models context trimming."""
    head, body, tail = _split(msgs)
    n_pairs = len(body) // 2
    drop = int(round(fraction * n_pairs))
    if drop <= 0:
        return msgs, {"breach": "truncate", "fraction": fraction, "pairs_dropped": 0}
    return head + body[2 * drop:] + tail, {
        "breach": "truncate", "fraction": fraction, "pairs_dropped": drop}


def breach_drop_pair(msgs, index=0, **kw):
    """Delete exactly one carried-forward pair. Predicted null for Omega (D2)."""
    head, body, tail = _split(msgs)
    n_pairs = len(body) // 2
    if n_pairs == 0:
        return msgs, {"breach": "drop_pair", "pairs_dropped": 0}
    i = index % n_pairs
    return head + body[:2 * i] + body[2 * i + 2:] + tail, {
        "breach": "drop_pair", "pair_index": i, "pairs_dropped": 1}


def breach_compact(msgs, k_fraction=0.4, summariser=None, **kw):
    """Replace the oldest k pairs with a single summary pair. Models compaction.

    `summariser` is a callable taking the list of (prompt, response) tuples and returning
    a summary string. It is passed in rather than imported so the breach is deterministic
    and testable without a model call.
    """
    head, body, tail = _split(msgs)
    n_pairs = len(body) // 2
    k = int(round(k_fraction * n_pairs))
    if k <= 0:
        return msgs, {"breach": "compact", "k_fraction": k_fraction, "pairs_replaced": 0}
    old = [(body[2 * i]["content"], body[2 * i + 1]["content"]) for i in range(k)]
    summary = summariser(old) if summariser else _naive_summary(old)
    replacement = [{"role": "user", "content": "[earlier conversation]"},
                   {"role": "assistant", "content": summary}]
    return head + replacement + body[2 * k:] + tail, {
        "breach": "compact", "k_fraction": k_fraction, "pairs_replaced": k,
        "summary_chars": len(summary)}


def _naive_summary(pairs):
    """Deterministic stand-in: first sentence of each response, concatenated."""
    out = []
    for _, r in pairs:
        s = r.strip().split(". ")
        out.append(s[0].strip() if s else "")
    return ". ".join(x for x in out if x) + "."


def breach_rewrite(msgs, rewriter=None, scope="all", **kw):
    """Rewrite carried-forward content in transit, preserving length where the rewriter
    does. Models guardrail / normalisation middleware. Predicted null for Omega if entropy
    is preserved (D5a)."""
    head, body, tail = _split(msgs)
    f = rewriter or (lambda s: s)
    n_pairs = len(body) // 2
    idxs = range(n_pairs) if scope == "all" else [int(scope) % max(n_pairs, 1)]
    new = copy.deepcopy(body)
    for i in idxs:
        new[2 * i + 1]["content"] = f(new[2 * i + 1]["content"])
    return head + new + tail, {"breach": "rewrite", "scope": scope,
                               "pairs_rewritten": len(list(idxs))}


BREACHES = {
    "none": breach_none,
    "truncate": breach_truncate,
    "drop_pair": breach_drop_pair,
    "compact": breach_compact,
    "rewrite": breach_rewrite,
}


# ---------------------------------------------------------------------------
# the recorder
# ---------------------------------------------------------------------------

class SeamRecorder:
    """Writes one JSONL record per turn: the emitted pair, and the carried-forward
    region as delivered. No entropy, no Omega."""

    def __init__(self, path, tok, run_id, condition, params,
                 K=K_DECLARED, W=W_DEFAULT, n_min=N_MIN):
        self.f = open(path, "w", encoding="utf-8")
        self.tok = tok
        self.f.write(json.dumps({
            "record": "manifest", "schema": SCHEMA_VERSION,
            "run_id": run_id, "condition": condition, "params": params,
            "K_declared": K, "W": W, "n_min": n_min,
            "tokenizer": getattr(tok, "name", "unknown"),
            "compared_region": "prior (prompt, response) pairs only; "
                               "system material and the current prompt excluded",
        }) + "\n")

    def record_turn(self, turn, emitted_prompt, emitted_response,
                    delivered_pairs, breach_meta):
        """emitted_*  : the pair as sent / returned this turn        (Z side)
           delivered_pairs : carried_forward(msgs) AFTER any breach  (S side)"""
        zp, zpn = sparse_counts(emitted_prompt, self.tok)
        zr, zrn = sparse_counts(emitted_response, self.tok)

        s_side = []
        for i, (p, r) in enumerate(delivered_pairs):
            cp, cpn = sparse_counts(p, self.tok)
            cr, crn = sparse_counts(r, self.tok)
            s_side.append({"slot": i,
                           "q_prompt": cp, "n_prompt": cpn,
                           "q_response": cr, "n_response": crn,
                           "eligible": (cpn + crn) >= N_MIN})

        self.f.write(json.dumps({
            "record": "turn", "turn": turn,
            "breach": breach_meta,
            "Z": {"q_prompt": zp, "n_prompt": zpn,
                  "q_response": zr, "n_response": zrn,
                  "eligible": (zpn + zrn) >= N_MIN},
            "S": s_side,
            "n_delivered_pairs": len(delivered_pairs),
        }) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------
# the call-site pattern, for reference
# ---------------------------------------------------------------------------

CALL_SITE = """
    # --- T1 instrumentation, at the answering-model call site -----------------
    answering_msgs.append({"role": "user", "content": prompt})

    breached, meta = BREACHES[CONDITION](answering_msgs, **BREACH_PARAMS)

    response, ans_read, _ = chat(ANSWERING_MODEL, breached, SEED + turn)

    rec.record_turn(turn, prompt, response, carried_forward(breached), meta)

    answering_msgs.append({"role": "assistant", "content": response})
    # NOTE: the UNBREACHED list carries forward, so each breach is applied fresh
    # in transit rather than accumulating in the application's own history.
"""

if __name__ == "__main__":
    # structural self-test, no model calls
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(1, 6):
        msgs += [{"role": "user", "content": f"p{i}"},
                 {"role": "assistant", "content": f"r{i} alpha beta gamma delta"}]
    msgs += [{"role": "user", "content": "p6"}]

    print("pairs in carried-forward region:", len(carried_forward(msgs)), "(expect 5)")
    for name, kw in [("none", {}), ("truncate", {"fraction": 0.4}),
                     ("drop_pair", {"index": 2}), ("compact", {"k_fraction": 0.4}),
                     ("rewrite", {"rewriter": str.upper})]:
        out, meta = BREACHES[name](msgs, **kw)
        print(f"  {name:10s} -> {len(carried_forward(out))} pairs   {meta}")
        assert out[0]["role"] == "system", "system message must survive"
        assert out[-1]["content"] == "p6", "current prompt must survive"
    print("\nsystem message and current prompt preserved by every breach: ok")
