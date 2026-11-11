"""Causal attention and the online-softmax (FlashAttention-style) formulation in plain Python."""
import math

def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [v / s for v in e]

def matvec(W, x): return [sum(w * v for w, v in zip(row, x)) for row in W]

def causal_attention(X, Wq, Wk, Wv):
    """X: T vectors of size d. Position t only sees positions <= t."""
    d = len(Wq)
    Q = [matvec(Wq, x) for x in X]; K = [matvec(Wk, x) for x in X]; V = [matvec(Wv, x) for x in X]
    out = []
    for t in range(len(X)):
        scores = [sum(a * b for a, b in zip(Q[t], K[s])) / math.sqrt(d) for s in range(t + 1)]
        w = softmax(scores)
        out.append([sum(w[s] * V[s][j] for s in range(t + 1)) for j in range(len(V[0]))])
    return out

def attention_ref(q, K, V):
    s = [sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q)) for k in K]
    m = max(s); e = [math.exp(x - m) for x in s]; z = sum(e)
    return [sum(e[i] / z * V[i][j] for i in range(len(K))) for j in range(len(V[0]))]

def attention_online(q, K, V, block=2):
    m, l, o = -1e30, 0.0, [0.0] * len(V[0])
    for start in range(0, len(K), block):
        ks, vs = K[start:start + block], V[start:start + block]
        s = [sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q)) for k in ks]
        m2 = max(m, max(s)); scale = math.exp(m - m2)
        p = [math.exp(x - m2) for x in s]
        l = l * scale + sum(p)
        o = [oi * scale + sum(p[i] * vs[i][j] for i in range(len(ks))) for j, oi in enumerate(o)]
        m = m2
    return [x / l for x in o]
