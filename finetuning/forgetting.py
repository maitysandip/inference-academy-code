import math, random
random.seed(21)
D = 12
wA = [random.gauss(0, 1) if i < 7 else 0.0 for i in range(D)]            # task A uses features 0-6
wB = [random.gauss(0, 1) if 4 <= i < 11 else 0.0 for i in range(D)]      # task B uses features 4-10 (overlap 4-6)
def sample(w, n):
    X = [[random.gauss(0, 1) for _ in range(D)] for _ in range(n)]
    y = [1 if sum(a*b for a, b in zip(w, x)) > 0 else 0 for x in X]
    return X, y
sig = lambda z: 1/(1+math.exp(-max(min(z, 30), -30)))
def acc(w, X, y): return sum((sum(a*b for a, b in zip(w, x)) > 0) == (t == 1) for x, t in zip(X, y))/len(X)
def sgd(w, data, lr, steps, batch=16, anchor=None, lam=0.0):
    w = w[:]
    for _ in range(steps):
        b = random.sample(data, batch); g = [0.0]*D
        for x, t in b:
            p = sig(sum(a*c for a, c in zip(w, x)))
            for k in range(D): g[k] += (p - t)*x[k]
        w = [wk - lr*(gk/batch + (lam*(wk - ak) if anchor else 0.0)) for wk, gk, ak in zip(w, g, anchor or w)]
    return w
XA, yA = sample(wA, 600); XB, yB = sample(wB, 600)
tA = list(zip(XA, yA)); tB = list(zip(XB, yB))
XAt, yAt = sample(wA, 2000); XBt, yBt = sample(wB, 2000)
base = sgd([0.0]*D, tA, 0.5, 400)
print(f"base model after learning task A:   task A {acc(base, XAt, yAt):.1%}   task B {acc(base, XBt, yBt):.1%}")
rows = []
def report(name, w): rows.append((name, acc(w, XAt, yAt), acc(w, XBt, yBt)))
random.seed(5); report("fine-tune on B only, 400 steps",              sgd(base, tB, 0.5, 400))
random.seed(5); report("... only 20 steps (stop early)",              sgd(base, tB, 0.5, 20))
random.seed(5); report("... small learning rate (0.02), 400 steps",   sgd(base, tB, 0.02, 400))
random.seed(5); report("... mix in 25% task A examples (replay)",     sgd(base, tB[:450] + random.sample(tA, 150), 0.5, 400))
random.seed(5); report("... mix in 50% task A examples (replay)",     sgd(base, tB[:300] + random.sample(tA, 300), 0.5, 400))
random.seed(5); report("... anchor to base weights (penalty 0.5)",    sgd(base, tB, 0.5, 400, anchor=base, lam=0.5))
random.seed(5); report("... anchor to base weights (penalty 2.0)",    sgd(base, tB, 0.5, 400, anchor=base, lam=2.0))
print(f"{'method':<44}{'task A (old)':>13}{'task B (new)':>14}")
for n, a, b in rows: print(f"{n:<44}{a:>12.1%}{b:>14.1%}")

random.seed(5); tuned = sgd(base, tB, 0.5, 400)
print("\ninterpolating between base and fine-tuned weights: w = base + s * (tuned - base)")
print(f"{'s':>5}{'task A':>10}{'task B':>10}")
for s in (0, 0.25, 0.5, 0.65, 0.75, 1):
    w = [b + s*(t - b) for b, t in zip(base, tuned)]
    print(f"{s:>5}{acc(w, XAt, yAt):>10.1%}{acc(w, XBt, yBt):>10.1%}")
