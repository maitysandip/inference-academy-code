"""Percentiles, bootstrap CIs and the closed-loop vs open-loop (coordinated omission) demonstration."""
import math, random

def percentile(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100
    f = math.floor(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def bootstrap_ci(xs, stat, n=1000, seed=0, alpha=0.05):
    rng = random.Random(seed)
    vals = sorted(stat([rng.choice(xs) for _ in xs]) for _ in range(n))
    return vals[int(n * alpha / 2)], vals[int(n * (1 - alpha / 2)) - 1]

def service_times(n, seed, stall_every=200, stall=2.0, base=0.05):
    rng = random.Random(seed)
    return [stall if i % stall_every == stall_every - 1 else rng.expovariate(1 / base) for i in range(n)]

def open_loop(service, rate, seed=1):
    rng = random.Random(seed); t = free = 0.0; lat = []
    for s in service:
        t += rng.expovariate(rate)
        start = max(t, free); free = start + s
        lat.append(free - t)
    return lat

if __name__ == "__main__":
    svc = service_times(4000, 7)
    for name, xs in (("closed", svc), ("open", open_loop(svc, rate=12))):
        print(name, [round(percentile(xs, p), 3) for p in (50, 95, 99)])
