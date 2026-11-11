"""A tiny Llama-style inference engine in pure Python: RMSNorm, RoPE, grouped-query attention, SwiGLU, KV cache."""
import math, random


class Config:
    vocab, d, n_layer, n_head, n_kv, hidden, max_seq = 40, 16, 2, 4, 2, 24, 64

    @property
    def head_dim(self):
        return self.d // self.n_head


def mat(rng, r, c, std=0.15):
    return [[rng.gauss(0, std) for _ in range(c)] for _ in range(r)]


def init_weights(cfg, seed=0):
    rng = random.Random(seed)
    hd = cfg.head_dim
    W = {"emb": mat(rng, cfg.vocab, cfg.d), "out": mat(rng, cfg.vocab, cfg.d), "norm": [1.0] * cfg.d, "layers": []}
    for _ in range(cfg.n_layer):
        W["layers"].append({
            "wq": mat(rng, cfg.n_head * hd, cfg.d), "wk": mat(rng, cfg.n_kv * hd, cfg.d), "wv": mat(rng, cfg.n_kv * hd, cfg.d),
            "wo": mat(rng, cfg.d, cfg.n_head * hd), "gate": mat(rng, cfg.hidden, cfg.d), "up": mat(rng, cfg.hidden, cfg.d),
            "down": mat(rng, cfg.d, cfg.hidden), "n1": [1.0] * cfg.d, "n2": [1.0] * cfg.d})
    return W


def matvec(M, x):
    return [sum(a * b for a, b in zip(row, x)) for row in M]


def rmsnorm(x, g, eps=1e-5):
    s = (sum(v * v for v in x) / len(x) + eps) ** -0.5
    return [v * s * gi for v, gi in zip(x, g)]


def softmax(xs):
    m = max(xs)
    e = [math.exp(v - m) for v in xs]
    s = sum(e)
    return [v / s for v in e]


def silu(z):
    return z / (1 + math.exp(-z))


def rope(vec, pos, hd, base=10000.0):
    out = vec[:]
    for h in range(0, len(vec), hd):
        for i in range(0, hd, 2):
            th = pos * base ** (-i / hd)
            c, s = math.cos(th), math.sin(th)
            a, b = vec[h + i], vec[h + i + 1]
            out[h + i], out[h + i + 1] = a * c - b * s, a * s + b * c
    return out


class KVCache:
    def __init__(self, cfg):
        self.k = [[] for _ in range(cfg.n_layer)]
        self.v = [[] for _ in range(cfg.n_layer)]

    def __len__(self):
        return len(self.k[0])


def forward_token(cfg, W, token, pos, cache):
    hd, group = cfg.head_dim, cfg.n_head // cfg.n_kv
    x = W["emb"][token][:]
    for l, L in enumerate(W["layers"]):
        h = rmsnorm(x, L["n1"])
        q = rope(matvec(L["wq"], h), pos, hd)
        k = rope(matvec(L["wk"], h), pos, hd)
        v = matvec(L["wv"], h)
        cache.k[l].append(k)                       # append this position's K and V to the cache
        cache.v[l].append(v)
        att = []
        for head in range(cfg.n_head):
            kvh = head // group                    # grouped-query: several query heads share one KV head
            qs = q[head * hd:(head + 1) * hd]
            scores = [sum(a * b for a, b in zip(qs, kk[kvh * hd:(kvh + 1) * hd])) / math.sqrt(hd) for kk in cache.k[l]]
            w = softmax(scores)
            att += [sum(w[t] * cache.v[l][t][kvh * hd + i] for t in range(len(w))) for i in range(hd)]
        x = [a + b for a, b in zip(x, matvec(L["wo"], att))]
        h = rmsnorm(x, L["n2"])
        ff = [silu(g) * u for g, u in zip(matvec(L["gate"], h), matvec(L["up"], h))]
        x = [a + b for a, b in zip(x, matvec(L["down"], ff))]
    return matvec(W["out"], rmsnorm(x, W["norm"]))


def argmax(xs):
    return max(range(len(xs)), key=lambda j: xs[j])


def generate(cfg, W, prompt, n_new, temperature=0.0, seed=0):
    rng = random.Random(seed)
    cache = KVCache(cfg)
    logits = None
    for pos, t in enumerate(prompt):               # prefill
        logits = forward_token(cfg, W, t, pos, cache)
    out = []
    for i in range(n_new):                         # decode
        if temperature == 0:
            tok = argmax(logits)
        else:
            tok = rng.choices(range(cfg.vocab), weights=softmax([l / temperature for l in logits]))[0]
        out.append(tok)
        logits = forward_token(cfg, W, tok, len(prompt) + i, cache)
    return out


if __name__ == "__main__":
    cfg = Config()
    W = init_weights(cfg)
    prompt = [3, 14, 15, 9, 2, 6]
    fast = generate(cfg, W, prompt, 10)
    seq = prompt[:]                                # reference: recompute from scratch at every step
    for _ in range(10):
        c = KVCache(cfg)
        for pos, t in enumerate(seq):
            lg = forward_token(cfg, W, t, pos, c)
        seq.append(argmax(lg))
    print("cache == recompute:", fast == seq[len(prompt):], fast)
    print("KV bytes/token (fp16):", 2 * cfg.n_layer * cfg.n_kv * cfg.head_dim * 2,
          "vs full MHA:", 2 * cfg.n_layer * cfg.n_head * cfg.head_dim * 2)
