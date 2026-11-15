import math, random
random.seed(11)
D, INFO = 30, 5
true_w = [random.gauss(0, 1.5) for _ in range(INFO)] + [0.0]*(D-INFO)
def sample(n):
    X, y = [], []
    for _ in range(n):
        x = [random.gauss(0, 1) for _ in range(D)]
        z = sum(w*v for w, v in zip(true_w, x))
        p = 1/(1+math.exp(-z)); lab = 1 if random.random() < p else 0
        if random.random() < 0.10: lab = 1 - lab          # 10% label noise
        X.append(x); y.append(lab)
    return X, y
Xtr, ytr = sample(60); Xva, yva = sample(600)
def loss(w, X, y):
    tot = 0
    for x, t in zip(X, y):
        z = sum(a*b for a, b in zip(w, x)); p = 1/(1+math.exp(-z)); p = min(max(p, 1e-9), 1-1e-9)
        tot += -(t*math.log(p) + (1-t)*math.log(1-p))
    return tot/len(X)
def acc(w, X, y): return sum((sum(a*b for a, b in zip(w, x)) > 0) == (t == 1) for x, t in zip(X, y))/len(X)
def train(lr, epochs, batch=10, wd=0.0, log_at=()):
    w = [0.0]*D; hist = {}
    idx = list(range(len(Xtr)))
    for e in range(1, epochs+1):
        random.shuffle(idx)
        for i in range(0, len(idx), batch):
            g = [0.0]*D
            for j in idx[i:i+batch]:
                z = sum(a*b for a, b in zip(w, Xtr[j])); p = 1/(1+math.exp(-z))
                for k in range(D): g[k] += (p - ytr[j])*Xtr[j][k]
            w = [wk - lr*(gk/len(idx[i:i+batch]) + wd*wk) for wk, gk in zip(w, g)]
        if e in log_at: hist[e] = (loss(w, Xtr, ytr), loss(w, Xva, yva), acc(w, Xva, yva))
    return hist
print("learning rate sweep: validation loss after 1, 10 and 100 epochs (train loss in brackets)")
print(f"{'lr':>7}   {'1 epoch':>16}   {'10 epochs':>16}   {'100 epochs':>16}")
for lr in (0.0003, 0.01, 0.1, 1.0, 20.0):
    random.seed(5); h = train(lr, 100, log_at=(1, 10, 100))
    print(f"{lr:>7}   " + "   ".join(f"{h[e][1]:6.3f} ({h[e][0]:5.3f})" for e in (1, 10, 100)))
print("\nepochs at a good learning rate (0.1): where validation turns upward")
random.seed(5); h = train(0.1, 400, log_at=(1, 2, 3, 5, 10, 20, 50, 100, 200, 400))
print(f"{'epoch':>6}  {'train loss':>10}  {'val loss':>9}  {'val acc':>8}")
for e, (tl, vl, va) in h.items(): print(f"{e:>6}  {tl:10.3f}  {vl:9.3f}  {va:8.1%}")
best = min(h, key=lambda e: h[e][1]); print("best validation loss at epoch", best)
print("\nweight decay at 400 epochs (lr 0.1):")
for wd in (0.0, 0.01, 0.1):
    random.seed(5); r = train(0.1, 400, wd=wd, log_at=(400,))[400]; print(f"  wd {wd:>5}: train {r[0]:.3f}  val {r[1]:.3f}  acc {r[2]:.1%}")
