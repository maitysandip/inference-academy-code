"""A complete tiny GPT in pure Python: autograd, attention, training, sampling. No dependencies."""
import math, random

random.seed(42)

# ---------- 1. Autograd on scalars ----------
class Value:
    __slots__ = ("data", "grad", "_prev", "_back")
    def __init__(self, data, prev=()):
        self.data, self.grad, self._prev, self._back = data, 0.0, prev, None
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def back():
            self.grad += out.grad; o.grad += out.grad
        out._back = back; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def back():
            self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._back = back; return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def back(): self.grad += k * self.data ** (k - 1) * out.grad
        out._back = back; return out
    def exp(self):
        e = math.exp(self.data); out = Value(e, (self,))
        def back(): self.grad += e * out.grad
        out._back = back; return out
    def log(self):
        out = Value(math.log(self.data), (self,))
        def back(): self.grad += out.grad / self.data
        out._back = back; return out
    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))
        def back(): self.grad += (out.data > 0) * out.grad
        out._back = back; return out
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o if isinstance(o, Value) else -o)
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Value) else 1.0 / o)
    def backward(self):
        order, seen = [], set()
        def visit(v):
            if id(v) in seen: return
            seen.add(id(v))
            for p in v._prev: visit(p)
            order.append(v)
        visit(self); self.grad = 1.0
        for v in reversed(order):
            if v._back: v._back()

# ---------- 2. Data and tokenizer (character level) ----------
NAMES = """emma olivia ava sophia isabella mia amelia harper evelyn abigail emily ella elizabeth camila luna sofia avery mila aria
scarlett penelope layla chloe victoria madison eleanor grace nora riley zoey hannah hazel lily ellie violet lillian zoe stella
aurora natalia addison lucy audrey brooklyn bella claire skylar lucia paisley everly anna caroline nova genesis emilia kennedy
maya willow kinsley naomi aaliyah elena sarah ariana allison gabriella alice madelyn cora ruby eva serenity autumn adeline hailey
gianna valentina isla eliana quinn nevaeh ivy sadie piper lydia alexa josephine emery julia delilah arianna vivian kaylee sophie
brielle madeline peyton rylee clara hadley melanie mackenzie reagan adalynn liliana aubree jade katherine isabelle natalie raelynn
maria athena ximena arya leilani taylor faith rose kylie alexandra mary margaret lyla ashley amaya eliza brianna bailey andrea""".split()
random.shuffle(NAMES)
chars = sorted(set("".join(NAMES)))
BOS = len(chars)                     # one special token marks start and end of a name
VOCAB = len(chars) + 1
stoi = {c: i for i, c in enumerate(chars)}

# ---------- 3. Parameters ----------
N_EMBD, N_HEAD, N_LAYER, BLOCK = 16, 4, 1, 12
HEAD_DIM = N_EMBD // N_HEAD

def matrix(r, c, std=0.08): return [[Value(random.gauss(0, std)) for _ in range(c)] for _ in range(r)]
P = {"wte": matrix(VOCAB, N_EMBD), "wpe": matrix(BLOCK, N_EMBD), "lm_head": matrix(VOCAB, N_EMBD)}
for l in range(N_LAYER):
    for n in ("wq", "wk", "wv", "wo"): P[f"l{l}.{n}"] = matrix(N_EMBD, N_EMBD)
    P[f"l{l}.fc1"] = matrix(4 * N_EMBD, N_EMBD); P[f"l{l}.fc2"] = matrix(N_EMBD, 4 * N_EMBD)
PARAMS = [p for m in P.values() for row in m for p in row]

# ---------- 4. Model ----------
def linear(x, w): return [sum((wi * xi for wi, xi in zip(row, x)), Value(0.0)) for row in w]
def softmax(xs):
    m = max(x.data for x in xs); e = [(x - m).exp() for x in xs]; s = sum(e, Value(0.0)); return [v / s for v in e]
def rmsnorm(x):
    ms = sum((xi * xi for xi in x), Value(0.0)) / len(x); scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token, pos, keys, values):
    x = [t + p for t, p in zip(P["wte"][token], P["wpe"][pos])]
    x = rmsnorm(x)
    for l in range(N_LAYER):
        h = rmsnorm(x)
        q, k, v = linear(h, P[f"l{l}.wq"]), linear(h, P[f"l{l}.wk"]), linear(h, P[f"l{l}.wv"])
        keys[l].append(k); values[l].append(v)                       # the KV cache grows by one position
        out = []
        for hd in range(N_HEAD):
            s = slice(hd * HEAD_DIM, (hd + 1) * HEAD_DIM)
            scores = [sum((a * b for a, b in zip(q[s], kk[s])), Value(0.0)) / HEAD_DIM ** 0.5 for kk in keys[l]]
            w = softmax(scores)
            out += [sum((w[t] * values[l][t][s][i] for t in range(len(w))), Value(0.0)) for i in range(HEAD_DIM)]
        x = [a + b for a, b in zip(x, linear(out, P[f"l{l}.wo"]))]   # residual
        h = rmsnorm(x)
        h = [u.relu() for u in linear(h, P[f"l{l}.fc1"])]
        x = [a + b for a, b in zip(x, linear(h, P[f"l{l}.fc2"]))]    # residual
    return linear(x, P["lm_head"])                                   # logits

# ---------- 5. Training ----------
def loss_for(name):
    toks = [BOS] + [stoi[c] for c in name] + [BOS]
    toks = toks[:BLOCK + 1]
    keys, values = [[] for _ in range(N_LAYER)], [[] for _ in range(N_LAYER)]
    losses = []
    for pos in range(len(toks) - 1):
        probs = softmax(gpt(toks[pos], pos, keys, values))
        losses.append(-probs[toks[pos + 1]].log())
    return sum(losses, Value(0.0)) / len(losses)

def train(steps=300, lr=0.05, log_every=50):
    m = [0.0] * len(PARAMS); v = [0.0] * len(PARAMS); b1, b2, eps = 0.9, 0.95, 1e-8
    history = []
    for step in range(1, steps + 1):
        name = NAMES[step % len(NAMES)]
        loss = loss_for(name)
        for p in PARAMS: p.grad = 0.0
        loss.backward()
        lr_t = lr * (1 - step / steps)                               # linear decay
        for i, p in enumerate(PARAMS):                               # Adam
            m[i] = b1 * m[i] + (1 - b1) * p.grad; v[i] = b2 * v[i] + (1 - b2) * p.grad ** 2
            mh, vh = m[i] / (1 - b1 ** step), v[i] / (1 - b2 ** step)
            p.data -= lr_t * mh / (vh ** 0.5 + eps)
        history.append(loss.data)
        if step % log_every == 0 or step == 1:
            print(f"step {step:4d} loss {loss.data:.3f}  (avg last {log_every}: {sum(history[-log_every:]) / len(history[-log_every:]):.3f})")
    return history

# ---------- 6. Sampling ----------
def sample(temperature=0.8, max_len=BLOCK):
    keys, values = [[] for _ in range(N_LAYER)], [[] for _ in range(N_LAYER)]
    tok, out = BOS, []
    for pos in range(max_len):
        probs = softmax([l / temperature for l in gpt(tok, pos, keys, values)])
        tok = random.choices(range(VOCAB), weights=[p.data for p in probs])[0]
        if tok == BOS: break
        out.append(chars[tok])
    return "".join(out)

if __name__ == "__main__":
    print(f"{len(NAMES)} names, vocab {VOCAB}, {len(PARAMS)} parameters")
    print(f"uniform-guess loss = ln({VOCAB}) = {math.log(VOCAB):.3f}")
    train()
    print("samples:", [sample() for _ in range(8)])
