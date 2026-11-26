"""Self-hosting versus an API: break-even volume, and which assumptions move it (standard library only).

  python3 break_even.py                  # the worked example: break-even curve and sensitivity
  python3 break_even.py --api-price 0.30 --gpu-price 1.80

Self-hosting cost is a step function of volume: you need enough GPUs for the average load
at a target utilisation, plus spares, and every GPU costs the same whether it is busy or not.
The API cost is a straight line. The break-even volume is where the line first rises above
the steps. That only happens if each extra GPU serves tokens more cheaply than the API:

  marginal self-host cost per million = gpu_price x (1 + overhead) / (gpu_tok_s x 3600 x utilisation) x 1e6

If that is above the API price, self-hosting never breaks even, however large the volume.
Sensitivity: change one input at a time by -30% and +30% and see how far the break-even
volume moves. The input that moves it most is the one to verify first.

Every number here is an illustrative input, not a market price: replace them with quotes.
"""
import argparse
import math

HOURS_PER_MONTH = 730
SECONDS_PER_MONTH = HOURS_PER_MONTH * 3600


def gpus_needed(tokens_per_month, gpu_tok_s, utilisation, spares):
    avg_tok_s = tokens_per_month / SECONDS_PER_MONTH
    return max(1, math.ceil(avg_tok_s / (gpu_tok_s * utilisation))) + spares


def self_host_monthly(tokens_per_month, p):
    gpus = gpus_needed(tokens_per_month, p["gpu_tok_s"], p["utilisation"], p["spares"])
    return gpus * p["gpu_price"] * HOURS_PER_MONTH * (1 + p["overhead"]) + p["people"], gpus


def api_monthly(tokens_per_month, p):
    return tokens_per_month / 1e6 * p["api_price"]


def marginal_cost_per_million(p):
    return p["gpu_price"] * (1 + p["overhead"]) / (p["gpu_tok_s"] * 3600 * p["utilisation"]) * 1e6


def break_even(p, lo=1e8, hi=1e13):
    """Smallest monthly volume (tokens) at which self-hosting is no dearer than the API.
    The self-host cost only steps up, and the API line rises steadily, so search the volume
    where the API cost first meets the self-host cost. Returns None if it never does."""
    if api_monthly(hi, p) < self_host_monthly(hi, p)[0]:
        return None
    for _ in range(200):                       # geometric bisection over a wide range
        mid = math.sqrt(lo * hi)
        if api_monthly(mid, p) >= self_host_monthly(mid, p)[0]:
            hi = mid
        else:
            lo = mid
    return hi


DEFAULTS = dict(
    api_price=0.60,      # $ per million blended tokens for a comparable model (illustrative)
    gpu_price=2.50,      # $ per GPU-hour (illustrative)
    gpu_tok_s=3000,      # tokens/s one GPU sustains at your SLO, measured
    utilisation=0.6,     # average busy share you can really achieve
    spares=1,            # N+1
    overhead=0.15,       # storage, network, monitoring as a share of GPU cost
    people=6000,         # $ per month of engineering and on-call attributable to the service
)


def fmt_tokens(t):
    return "never" if t is None else f"{t / 1e9:,.1f} B tokens/month ({t / SECONDS_PER_MONTH:,.0f} tok/s average)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    p = vars(ap.parse_args(argv))

    print("Inputs (illustrative, replace with your own):")
    for k, v in p.items():
        print(f"  {k:12} {v}")

    print("\nMonthly cost by volume")
    print(f"{'tokens/month':>14} {'API $':>10} {'self-host $':>12} {'GPUs':>5}  cheaper")
    for billions in (1, 10, 30, 50, 100, 200):
        t = billions * 1e9
        api = api_monthly(t, p)
        own, gpus = self_host_monthly(t, p)
        print(f"{billions:>12} B {api:>10,.0f} {own:>12,.0f} {gpus:>5}  {'API' if api < own else 'self-host'}")

    m = marginal_cost_per_million(p)
    print(f"\nMarginal self-host cost: ${m:.3f} per million tokens against the API's ${p['api_price']:.3f}"
          + ("" if m < p["api_price"] else ": above the API price, so self-hosting never breaks even"))
    base = break_even(p)
    print(f"Break-even: {fmt_tokens(base)}")

    print("\nSensitivity: break-even volume with one input changed by -30% and +30%")
    print(f"{'input':12} {'-30%':>12} {'+30%':>12}   swing")
    rows = []
    for k in ("api_price", "gpu_price", "gpu_tok_s", "utilisation", "people", "overhead"):
        out = []
        for f in (0.7, 1.3):
            q = dict(p)
            q[k] = p[k] * f
            if k == "utilisation":
                q[k] = min(q[k], 0.95)
            out.append(break_even(q))
        swing = (max(o for o in out if o) - min(o for o in out if o)) if all(out) else float("inf")
        rows.append((swing, k, out))
    for swing, k, (lo_v, hi_v) in sorted(rows, reverse=True):
        def b(v):
            return "never" if v is None else f"{v / 1e9:,.1f} B"
        print(f"{k:12} {b(lo_v):>12} {b(hi_v):>12}   {'inf' if swing == float('inf') else f'{swing / 1e9:,.1f} B'}")


if __name__ == "__main__":
    main()
