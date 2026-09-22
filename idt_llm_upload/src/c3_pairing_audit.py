"""
C3 reproduction audit — exactly how a PhotoBook game becomes (S, A, S') triples.

Emits, per game:
    msgs_raw        every message in every round
    msgs_nosel      after dropping <selection> messages
    utterances      after collapsing consecutive same-speaker messages
    pairs           the turn count used as n_turns  (this is the number we call
                    a "turn": one A -> B pair, consumed NON-OVERLAPPING)
    unpaired        utterances the pairing walk skipped
    first_speaker   agent_id of side A
    drop_reason     "" | "no_messages" | "score_anomaly" | "below_floor"

and writes c3_manifest.csv with one row per game, kept and dropped alike.
"""
import os, glob, json
import numpy as np
import pandas as pd

LOGS = "/tmp/pb/logs/logs"
FLOOR = 25
OUT = "/home/claude/eps/c3_audit"
os.makedirs(OUT, exist_ok=True)


def audit(path):
    g = json.load(open(path))
    raw = 0
    seq, score, flag4 = [], 0, False
    for r in g["rounds"]:
        s = r.get("score")
        if isinstance(s, dict):
            for v in s.values():
                if v > 3:
                    flag4 = True
            score += sum(s.values())
        for m in r["messages"]:
            raw += 1
            t = str(m["message"])
            if t.startswith("<selection>"):
                continue                       # an action, not language
            seq.append((str(m["agent_id"]), t))

    rec = dict(game=os.path.basename(path)[:-5], msgs_raw=raw,
               msgs_nosel=len(seq), rounds=len(g["rounds"]), score=score,
               score_anomaly=bool(flag4))
    if not seq:
        rec.update(utterances=0, pairs=0, unpaired=0, first_speaker="",
                   drop_reason="no_messages")
        return rec

    # collapse consecutive same-speaker messages into one utterance
    coll = []
    for spk, txt in seq:
        if coll and coll[-1][0] == spk:
            coll[-1][1].append(txt)
        else:
            coll.append([spk, [txt]])

    a = coll[0][0]                              # side A = first speaker
    pairs, unpaired, i = 0, 0, 0
    while i < len(coll) - 1:
        if coll[i][0] == a and coll[i + 1][0] != a:
            pairs += 1
            i += 2                              # NON-OVERLAPPING consumption
        else:
            unpaired += 1
            i += 1
    if i == len(coll) - 1:
        unpaired += 1                           # trailing utterance, never used

    reason = ""
    if flag4:
        reason = "score_anomaly"
    elif pairs < FLOOR:
        reason = "below_floor"
    rec.update(utterances=len(coll), pairs=pairs, unpaired=unpaired,
               first_speaker=a, drop_reason=reason)
    return rec


rows = [audit(f) for f in sorted(glob.glob(f"{LOGS}/*.json"))]
t = pd.DataFrame(rows)
t["kept"] = t.drop_reason == ""
t.to_csv(f"{OUT}/c3_manifest.csv", index=False)

print(f"games in logs           {len(t)}")
print(f"  no messages           {(t.drop_reason == 'no_messages').sum()}")
print(f"  score anomaly (>3)    {(t.drop_reason == 'score_anomaly').sum()}")
print(f"  below the {FLOOR} floor    {(t.drop_reason == 'below_floor').sum()}")
print(f"  KEPT                  {t.kept.sum()}")
print(f"  dropped total         {(~t.kept).sum()}")

k = t[t.kept]
print(f"\nkept games: pairs {k.pairs.min()}-{k.pairs.max()}   "
      f"utterances {k.utterances.min()}-{k.utterances.max()}   "
      f"messages(no sel) {k.msgs_nosel.min()}-{k.msgs_nosel.max()}   "
      f"messages(raw) {k.msgs_raw.min()}-{k.msgs_raw.max()}")
print(f"kept games where pairs == utterances // 2 exactly: "
      f"{int((k.pairs == k.utterances // 2).sum())} of {len(k)}")
print(f"unpaired utterances per kept game: {k.unpaired.min()}-{k.unpaired.max()}")

# cross-check against the published table
pb = pd.read_csv("/home/claude/eps/pb/pb_fullP.csv")
m = pb.merge(k[["game", "pairs", "first_speaker"]], on="game", how="left")
print(f"\ncross-check vs pb_fullP.csv: rows {len(pb)}   "
      f"n_turns == pairs in {int((m.n_turns == m.pairs).sum())} of {len(m)}")
print(f"games in pb_fullP not kept by this audit: {int(m.pairs.isna().sum())}")
print(f"\nwrote {OUT}/c3_manifest.csv")
