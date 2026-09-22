"""Per-message measurement primitive.

One message in -> token count, distinct-token count, Shannon entropy (bits),
and the same three quantities over adjacent token pairs (bigrams).

Nothing here uses meaning, context, or model internals.
"""
import math
from collections import Counter
from transformers import AutoTokenizer


def shannon_H(counts):
    """Shannon entropy in bits of a frequency histogram."""
    n = sum(counts.values())
    if n == 0:
        return 0.0
    total = 0.0
    for c in counts.values():
        p = c / n
        total -= p * math.log2(p)
    return total


class Tokenizer:
    def __init__(self, name_or_path):
        self.name = name_or_path
        self.tok = AutoTokenizer.from_pretrained(name_or_path)

    def ids(self, text):
        if text is None:
            return []
        s = str(text)
        if not s:
            return []
        return self.tok.encode(s, add_special_tokens=False)


def message_stats(ids):
    """All per-message quantities, from a token id sequence."""
    uni = Counter(ids)
    bi = Counter(zip(ids, ids[1:])) if len(ids) >= 2 else Counter()
    return {
        "tokens":  len(ids),
        "types":   len(uni),
        "H":       shannon_H(uni),
        "pairs":   sum(bi.values()),
        "types_2": len(bi),
        "H_2":     shannon_H(bi),
    }
