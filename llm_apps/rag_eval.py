"""A RAG evaluation loop: score retrieval separately from generation, then compare two retriever settings.

Retrieval metrics per question (the golden set lists which chunk ids contain the answer):
  recall@k  - did any relevant chunk make the top k?
  MRR       - 1 / rank of the first relevant chunk (0 if none in the top k)
Generation is scored with two cheap checks that need no judge model:
  grounded  - every cited chunk id was actually retrieved
  correct   - the answer contains the expected key fact
The retriever is a tiny keyword scorer so the whole loop runs offline; swap in your real search.
"""
import math
import re

CHUNKS = {
    "c1": "Refunds for digital courses are accepted within 7 days of purchase.",
    "c2": "A refund is not possible once more than 20 percent of lessons are opened.",
    "c3": "Duplicate purchases are refunded at any time after a short review.",
    "c4": "Yearly plans are billed once and save about two months compared with monthly billing.",
    "c5": "Team plans include an invoice with your company tax number.",
    "c6": "You can change your email address from the account page.",
    "c7": "Certificates are issued when every lesson and quiz in a course is complete.",
    "c8": "Course access continues for as long as your plan is active.",
}
GOLDEN = [
    {"q": "How long do I have to ask for a refund?", "relevant": {"c1"}, "fact": "7 days"},
    {"q": "I bought the same course twice, can I get money back?", "relevant": {"c3"}, "fact": "any time"},
    {"q": "Does a yearly plan cost less?", "relevant": {"c4"}, "fact": "two months"},
    {"q": "How do I get an invoice for my company?", "relevant": {"c5"}, "fact": "invoice"},
    {"q": "When do I receive a certificate?", "relevant": {"c7"}, "fact": "every lesson"},
    {"q": "What if I opened most of the lessons, can I still be refunded?", "relevant": {"c2"}, "fact": "20 percent"},
]
STOP = set("a an the i do of to for is are can my me how what when if in on with you your from once still get".split())


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def stem(w):
    return w[:5]  # crude stemming so "refund", "refunds", "refunded" match


def retrieve(q, k, use_stem):
    norm = stem if use_stem else (lambda w: w)
    qs = {norm(w) for w in words(q)}
    df = {}
    for t in CHUNKS.values():
        for w in {norm(x) for x in words(t)}:
            df[w] = df.get(w, 0) + 1
    scored = []
    for cid, t in CHUNKS.items():
        ws = {norm(x) for x in words(t)}
        score = sum(math.log(1 + len(CHUNKS) / df[w]) for w in qs & ws)
        scored.append((-score, cid))
    return [cid for s, cid in sorted(scored)[:k] if s < 0]


def fake_generate(q, ids):
    """Stand-in for the model: answers from the top chunk and cites it."""
    if not ids:
        return "I don't know.", []
    return CHUNKS[ids[0]], [ids[0]]


def evaluate(k, use_stem):
    rec = mrr = grounded = correct = 0
    misses = []
    for g in GOLDEN:
        ids = retrieve(g["q"], k, use_stem)
        hit = [i for i, c in enumerate(ids) if c in g["relevant"]]
        rec += bool(hit)
        mrr += 1 / (hit[0] + 1) if hit else 0
        ans, cites = fake_generate(g["q"], ids)
        grounded += all(c in ids for c in cites)
        correct += g["fact"] in ans
        if not hit:
            misses.append(g["q"])
    n = len(GOLDEN)
    return rec / n, mrr / n, grounded / n, correct / n, misses


if __name__ == "__main__":
    print(f"{'retriever':<22} recall@k   MRR  grounded  correct")
    for k, st in [(1, False), (3, False), (3, True)]:
        r, m, g, c, misses = evaluate(k, st)
        name = f"k={k}, {'stemmed' if st else 'exact words'}"
        print(f"{name:<22} {r:8.2f} {m:5.2f} {g:9.2f} {c:8.2f}")
        for q in misses:
            print(f"    miss: {q}")
