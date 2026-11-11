"""Byte-level BPE tokenizer: train, encode, decode."""
from collections import Counter

def get_pairs(ids):
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

class BPE:
    def __init__(self):
        self.merges = {}                                  # (a, b) -> new id, in training order
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size):
        ids = list(text.encode("utf-8"))
        for new_id in range(256, vocab_size):
            pairs = get_pairs(ids)
            if not pairs: break
            pair = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))     # most frequent, deterministic tie-break
            if pairs[pair] < 2: break
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return ids

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pairs = get_pairs(ids)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))   # earliest-learned merge first
            if pair not in self.merges: break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

