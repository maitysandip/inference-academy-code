"""Cost and latency as evaluation metrics, next to quality.

Two candidate configurations are run over the same 400 test cases (simulated here with a fixed seed).
We report quality with a confidence interval, latency percentiles (not the mean), cost per request,
and the number that usually decides: cost per SUCCESSFUL request. Then we check each against an SLO.
"""
import math
import random

SLO = {"p95_latency_s": 5.0, "min_quality": 0.80}


def simulate(n, p_ok, ttft_med, tps, out_mean, cost_in, cost_out, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        out = max(20, int(rng.gauss(out_mean, out_mean * 0.35)))
        ttft = ttft_med * math.exp(rng.gauss(0, 0.45))  # long right tail
        lat = ttft + out / tps
        inp = 2500
        cost = (inp * cost_in + out * cost_out) / 1e6
        rows.append({"ok": rng.random() < p_ok, "lat": lat, "cost": cost})
    return rows


def pct(xs, p):
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))
    return xs[i]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def report(name, rows):
    n, k = len(rows), sum(r["ok"] for r in rows)
    lo, hi = wilson(k, n)
    lats = [r["lat"] for r in rows]
    cost = sum(r["cost"] for r in rows)
    p95 = pct(lats, 95)
    ok = p95 <= SLO["p95_latency_s"] and lo >= SLO["min_quality"]
    print(f"{name:<12} quality {k/n:.3f} [{lo:.3f}, {hi:.3f}]  mean {sum(lats)/n:.2f}s  p50 {pct(lats,50):.2f}s  p95 {p95:.2f}s  p99 {pct(lats,99):.2f}s")
    print(f"{'':<12} ${cost/n*1000:.2f} per 1k requests   ${cost/k*1000:.2f} per 1k successful   meets SLO: {ok}")


if __name__ == "__main__":
    print(f"SLO: p95 <= {SLO['p95_latency_s']} s and quality lower bound >= {SLO['min_quality']}\n")
    report("large model", simulate(400, 0.88, 0.9, 45, 350, 2.00, 8.00, seed=1))
    report("small model", simulate(400, 0.82, 0.35, 140, 380, 0.10, 0.40, seed=2))
    report("small + RAG", simulate(400, 0.86, 0.5, 140, 330, 0.10, 0.40, seed=3))
