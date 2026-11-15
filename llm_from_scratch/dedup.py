"""A pretraining data pipeline in miniature: clean, filter, exact-dedup and near-dedup with MinHash.

Web text is full of boilerplate, near-copies (the same article on ten mirror sites, with a different
footer each time) and junk. Training on duplicates wastes compute and makes models memorise. Steps:
  1. normalise (whitespace, case for hashing only)
  2. quality filters (length, symbol ratio, repeated lines)
  3. exact dedup by hash of the normalised text
  4. near-dedup: MinHash signatures over word 5-gram shingles, then LSH banding to find candidate pairs
"""
import hashlib
import random
import re

DOCS = {
    "a": "The KV cache stores keys and values for previous tokens so decoding does not recompute them. It grows with context length.",
    "b": "The KV cache stores keys and values for previous tokens so decoding does not recompute them. It grows with context length.",
    "c": "The KV cache stores keys and values for previous tokens, so decoding does not recompute them. It grows with the context length. Share this article!",
    "d": "Memory bandwidth decides decode speed because every generated token reads all active weights once from memory.",
    "e": "$$$ CLICK HERE $$$ !!! ### $$$ WIN ### !!! $$$",
    "f": "ok",
    "g": "Buy now. Buy now. Buy now. Buy now. Buy now. Buy now.",
    "h": "Memory bandwidth decides decode speed, since every generated token must read all of the active weights from memory once.",
}


def normalise(t):
    return re.sub(r"\s+", " ", t).strip()


def quality(t):
    words = t.split()
    if len(words) < 8:
        return "too short"
    sym = sum(not (c.isalnum() or c.isspace() or c in ".,;:'-()") for c in t) / len(t)
    if sym > 0.1:
        return f"symbol ratio {sym:.2f}"
    sents = [s.strip() for s in t.split(".") if s.strip()]
    if len(sents) >= 3 and len(set(sents)) / len(sents) < 0.5:
        return "repeated sentences"
    return None


def shingles(t, k=5):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def minhash(sh, n=64, seed=0):
    r = random.Random(seed)
    salts = [r.getrandbits(32) for _ in range(n)]
    return [min(int.from_bytes(hashlib.blake2b(f"{s}|{x}".encode(), digest_size=8).digest(), "big") for x in sh) for s in salts]


def jaccard(a, b):
    return len(a & b) / len(a | b)


def lsh_candidates(sigs, bands=16):
    rows = len(next(iter(sigs.values()))) // bands
    buckets, pairs = {}, set()
    for doc, sig in sigs.items():
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            for other in buckets.get(key, []):
                pairs.add(tuple(sorted((doc, other))))
            buckets.setdefault(key, []).append(doc)
    return pairs


if __name__ == "__main__":
    docs = {k: normalise(v) for k, v in DOCS.items()}
    print(f"start: {len(docs)} documents")
    for k in list(docs):
        why = quality(docs[k])
        if why:
            print(f"  filter {k}: {why}")
            del docs[k]
    seen, exact = {}, []
    for k in list(docs):
        h = hashlib.sha256(docs[k].lower().encode()).hexdigest()
        if h in seen:
            exact.append((k, seen[h]))
            del docs[k]
        else:
            seen[h] = k
    print("exact duplicates removed:", exact)
    sh = {k: shingles(v) for k, v in docs.items()}
    sigs = {k: minhash(s) for k, s in sh.items()}
    print("\ncandidate near-duplicate pairs from LSH (16 bands x 4 rows):")
    drop = set()
    for a, b in sorted(lsh_candidates(sigs)):
        est = sum(x == y for x, y in zip(sigs[a], sigs[b])) / len(sigs[a])
        true = jaccard(sh[a], sh[b])
        dup = est >= 0.5
        print(f"  {a}-{b}: MinHash estimate {est:.2f}, true Jaccard {true:.2f} -> {'near-duplicate, drop ' + b if dup else 'keep both'}")
        if dup:
            drop.add(b)
    for k in drop:
        del docs[k]
    print("\nkept:", sorted(docs))
    print("pairs never compared (no shared band):", sorted({tuple(sorted((a, b))) for a in sh for b in sh if a < b} - lsh_candidates(sigs)))
