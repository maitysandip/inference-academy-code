import math, random, sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
random.seed(SEED)
DIM, TAU = 16, 0.2
GROUPS, PER_GROUP = 4, 3                                       # 12 concepts; the 3 in a group are easy to confuse
concepts = [(g, c) for g in range(GROUPS) for c in range(PER_GROUP)]
def words(prefix, g, c, n): return [f"{prefix}{g}_{c}_{i}" for i in range(n)]
Q = {k: words("q", *k, 6) for k in concepts}                  # words customers use
D = {k: words("d", *k, 6) for k in concepts}                  # words the documentation uses (a different vocabulary)
SHARED = {g: [f"grp{g}_{i}" for i in range(3)] for g in range(GROUPS)}
GENERIC = [f"gen{i}" for i in range(6)]
def make_query(k): return random.sample(Q[k], 2) + random.sample(SHARED[k[0]], 2) + [random.choice(GENERIC)]
def make_doc(k):   return random.sample(D[k], 3) + random.sample(SHARED[k[0]], 2) + random.sample(GENERIC, 2)
train = [(k, make_query(k), make_doc(k)) for k in concepts for _ in range(25)]
test_docs = [(k, make_doc(k)) for k in concepts for _ in range(20)]
test_q = [(k, make_query(k)) for k in concepts for _ in range(15)]

def new_vectors(): return {w: [random.gauss(0, 0.3) for _ in range(DIM)] for w in sorted({w for _, q, d in train for w in q + d})}
def embed(E, ws):
    v = [0.0]*DIM
    for w in ws:
        for i in range(DIM): v[i] += E[w][i]
    return [x/len(ws) for x in v]
def evaluate(E):
    de = [embed(E, d) for _, d in test_docs]; mrr = r1 = 0
    for k, q in test_q:
        qe = embed(E, q); order = sorted(range(len(de)), key=lambda j: -sum(a*b for a, b in zip(qe, de[j])))
        rank = next(r for r, j in enumerate(order, 1) if test_docs[j][0] == k); mrr += 1/rank; r1 += rank == 1
    return r1/len(test_q), mrr/len(test_q)
def lexical():
    mrr = r1 = 0
    for k, q in test_q:
        order = sorted(range(len(test_docs)), key=lambda j: -len(set(q) & set(test_docs[j][1])))
        rank = next(r for r, j in enumerate(order, 1) if test_docs[j][0] == k); mrr += 1/rank; r1 += rank == 1
    return r1/len(test_q), mrr/len(test_q)

def fit(hard, epochs=12, lr=0.5, batch=24):
    E = new_vectors(); by_group = {}
    for k, q, d in train: by_group.setdefault(k[0], []).append((k, d))
    for _ in range(epochs):
        random.shuffle(train)
        for s in range(0, len(train) - batch + 1, batch):
            b = train[s:s+batch]
            qs = [q for _, q, _ in b]; ds = [d for _, _, d in b]
            if hard:   # one hard negative per query: a document of a different concept in the same group
                ds = ds + [random.choice([x for x in by_group[k[0]] if x[0] != k])[1] for k, _, _ in b]
            qe = [embed(E, q) for q in qs]; de = [embed(E, d) for d in ds]
            gq = [[0.0]*DIM for _ in qs]; gd = [[0.0]*DIM for _ in ds]
            for i in range(len(qs)):
                sc = [sum(a*c for a, c in zip(qe[i], de[j]))/TAU for j in range(len(ds))]
                m = max(sc); ex = [math.exp(x-m) for x in sc]; z = sum(ex)
                for j in range(len(ds)):
                    g = ex[j]/z - (1.0 if j == i else 0.0)
                    for t in range(DIM): gq[i][t] += g*de[j][t]/TAU; gd[j][t] += g*qe[i][t]/TAU
            for ws, g in list(zip(qs, gq)) + list(zip(ds, gd)):
                for w in ws:
                    for t in range(DIM): E[w][t] -= lr*g[t]/len(ws)/len(qs)
    return E
l1, lm = lexical(); print(f"{'method':<34}{'recall@1':>9}{'MRR':>8}")
print(f"{'keyword overlap (no training)':<34}{l1:>9.1%}{lm:>8.3f}")
E0 = new_vectors(); r1, m = evaluate(E0); print(f"{'random embeddings (untrained)':<34}{r1:>9.1%}{m:>8.3f}")
random.seed(SEED); r1, m = evaluate(fit(hard=False)); print(f"{'trained, in-batch negatives':<34}{r1:>9.1%}{m:>8.3f}")
random.seed(SEED); r1, m = evaluate(fit(hard=True)); print(f"{'trained, plus hard negatives':<34}{r1:>9.1%}{m:>8.3f}")
