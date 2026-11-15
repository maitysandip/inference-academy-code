"""Cost and latency budgets for an LLM feature, per request and per month.

Latency of one model call  ~= TTFT + output_tokens / decode_speed.
An agent or chain makes several calls in sequence, so its latency is the sum.
Cost per call = input tokens x input price + output tokens x output price, with cached input billed at a discount.
Prices are illustrative placeholders (USD per million tokens); put in your provider's current numbers.
"""

PRICES = {  # (input, cached input, output) per 1M tokens: illustrative only
    "small": (0.10, 0.025, 0.40),
    "large": (2.00, 0.50, 8.00),
}


def call(model, inp, out, cached=0, ttft=0.4, tps=60):
    pi, pc, po = PRICES[model]
    cost = ((inp - cached) * pi + cached * pc + out * po) / 1e6
    return {"model": model, "cost": cost, "latency": ttft + out / tps}


def plan(name, calls, requests_per_month):
    cost = sum(c["cost"] for c in calls)
    lat = sum(c["latency"] for c in calls)
    print(f"{name:<34} {len(calls)} calls  {lat:5.1f} s  ${cost*1000:7.3f} per 1k requests  ${cost*requests_per_month:9,.0f} per month")
    return cost, lat


if __name__ == "__main__":
    N = 300_000  # requests per month
    print(f"{'design':<34} calls  latency   cost")
    plan("one large call", [call("large", 3000, 400)], N)
    plan("one large call, 2.5k cached prefix", [call("large", 3000, 400, cached=2500)], N)
    plan("small router + large answer", [call("small", 600, 10, tps=150), call("large", 3000, 400, cached=2500)], N)
    plan("small router + small answer", [call("small", 600, 10, tps=150), call("small", 3000, 400, cached=2500, tps=150)], N)
    agent = [call("large", 3000 + 900 * k, 150, cached=2500 + 900 * k) for k in range(5)] + [call("large", 7500, 400, cached=7000)]
    plan("agent: 5 tool steps + final answer", agent, N)

    print("\nBlended routing (70% small, 30% large):")
    small = sum(c["cost"] for c in [call("small", 600, 10, tps=150), call("small", 3000, 400, cached=2500, tps=150)])
    large = sum(c["cost"] for c in [call("small", 600, 10, tps=150), call("large", 3000, 400, cached=2500)])
    blended = 0.7 * small + 0.3 * large
    print(f"  ${blended*1000:.3f} per 1k requests, ${blended*N:,.0f} per month")

    print("\nLatency budget for a 3 s p50 target with one large call:")
    for out in (200, 400, 800):
        need = out / (3.0 - 0.4)
        print(f"  {out} output tokens needs >= {need:.0f} tokens/s after a 0.4 s TTFT")
