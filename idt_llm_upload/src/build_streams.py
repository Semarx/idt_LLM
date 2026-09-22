"""Build the message table and the token-sequence store.

Reads a turns CSV (one row per turn, columns conv_id / Turn / Prompt / Response)
and writes:

  Data/messages.csv        one row per message, with all per-message quantities
  Data/token_seqs.jsonl    msg_id -> token id sequence (for order-based work later)

Tokenization happens once, here. Everything downstream reads these two files.
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from msg_entropy import Tokenizer, message_stats

COLS = ["msg_id", "conv_id", "platform", "title", "turn", "seq", "role",
        "tokens", "types", "H", "pairs", "types_2", "H_2"]


def build(turns_csv, tokenizer_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(turns_csv, dtype={"conv_id": str, "Turn": int},
                     keep_default_na=False)
    df = df.sort_values(["conv_id", "Turn"]).reset_index(drop=True)
    tk = Tokenizer(tokenizer_path)

    rows, t0 = [], time.time()
    seq_path = os.path.join(out_dir, "token_seqs.jsonl")
    with open(seq_path, "w") as fseq:
        for k, (cid, g) in enumerate(df.groupby("conv_id", sort=False)):
            g = g.sort_values("Turn")
            platform = g.platform.iloc[0] if "platform" in g else ""
            title = g.title.iloc[0] if "title" in g else ""
            seq = 0
            for r in g.itertuples():
                for role, text in (("prompt", r.Prompt), ("response", r.Response)):
                    seq += 1
                    ids = tk.ids(text)
                    msg_id = f"{cid}:{seq:06d}"
                    st = message_stats(ids)
                    rows.append({"msg_id": msg_id, "conv_id": cid,
                                 "platform": platform, "title": title,
                                 "turn": int(r.Turn), "seq": seq, "role": role, **st})
                    fseq.write(json.dumps({"msg_id": msg_id, "ids": ids}) + "\n")
            if (k + 1) % 10 == 0:
                print(f"  {k+1} conversations  {time.time()-t0:.0f}s", flush=True)

    M = pd.DataFrame(rows)[COLS]
    msg_path = os.path.join(out_dir, "messages.csv")
    M.to_csv(msg_path, index=False)
    print(f"\nmessages: {len(M)}  conversations: {M.conv_id.nunique()}  "
          f"{time.time()-t0:.0f}s")
    print(f"  {msg_path}")
    print(f"  {seq_path}")
    return M


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default="Data")
    a = ap.parse_args()
    build(a.turns, a.tokenizer, a.out)
