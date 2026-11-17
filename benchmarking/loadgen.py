"""Open-loop load generator for an OpenAI-compatible streaming endpoint (stdlib only)."""
import argparse, json, math, random, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

def percentile(xs, p):
    xs = sorted(xs)
    if not xs: return float("nan")
    k = (len(xs) - 1) * p / 100; f = math.floor(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def one_request(url, model, prompt, max_tokens, intended_t0, results, timeout=120):
    body = json.dumps({"model": model, "stream": True, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    first = None; n = 0; err = None; last = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                now = time.perf_counter()
                if first is None: first = now
                last = now; n += 1
    except Exception as e:
        err = str(e)
    end = time.perf_counter()
    results.append({"ttft": (first - intended_t0) if first else None, "e2e": end - intended_t0,
                    "tokens": n, "tpot": ((last - first) / max(n - 1, 1)) if first and n > 1 else None, "err": err})

def run(url, model, rate, duration, warmup, prompt_lens, out_lens, seed=0, workers=256):
    rng = random.Random(seed); results = []; t_start = time.perf_counter(); t = 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while t < duration:
            t += rng.expovariate(rate)
            delay = t_start + t - time.perf_counter()
            if delay > 0: time.sleep(delay)
            intended = t_start + t                       # measure latency from the scheduled arrival time
            prompt = "word " * rng.choice(prompt_lens)
            pool.submit(one_request, url, model, prompt, rng.choice(out_lens), intended, results)
    return [r for r in results]

def summarise(results, slo_ttft, slo_tpot, duration, warmup):
    ok = [r for r in results if r["err"] is None and r["ttft"] is not None]
    ttft = [r["ttft"] for r in ok]; tpot = [r["tpot"] for r in ok if r["tpot"] is not None]
    good = [r for r in ok if r["ttft"] <= slo_ttft and (r["tpot"] is None or r["tpot"] <= slo_tpot)]
    span = max(duration - warmup, 1e-9)
    return {"requests": len(results), "errors": len(results) - len(ok),
            "ttft_p50": percentile(ttft, 50), "ttft_p95": percentile(ttft, 95), "ttft_p99": percentile(ttft, 99),
            "tpot_p95": percentile(tpot, 95), "tok_per_s": sum(r["tokens"] for r in ok) / span,
            "goodput_req_per_s": len(good) / span}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="model")
    ap.add_argument("--rates", default="2,4,8")
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--slo-ttft", type=float, default=1.0)
    ap.add_argument("--slo-tpot", type=float, default=0.06)
    a = ap.parse_args()
    for rate in [float(x) for x in a.rates.split(",")]:
        res = run(a.url, a.model, rate, a.duration, 0, [200, 800, 2000], [64, 200, 400])
        print(rate, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in summarise(res, a.slo_ttft, a.slo_tpot, a.duration, 0).items()}))
