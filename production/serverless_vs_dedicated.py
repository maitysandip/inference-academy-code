"""Serverless GPU versus a dedicated GPU: where the cost lines cross, and what cold starts do to latency.

Serverless bills per second the container is up (including warm-up and an idle window after the last
request); a dedicated GPU bills every hour. Prices are illustrative placeholders: use your provider's.
We model traffic as requests per hour arriving in bursts, each request taking busy_s seconds of GPU time.
"""
DEDICATED_PER_HOUR = 2.50      # USD, one GPU, always on
SERVERLESS_PER_SECOND = 0.0011  # USD, same GPU class, billed while the container is up (~3.96/h)
COLD_START_S = 45              # pull image + load weights, billed and added to the first request
IDLE_WINDOW_S = 60             # container stays warm this long after the last request
BUSY_S = 4.0                   # GPU seconds per request (batched; illustrative)


def serverless_cost_per_hour(req_per_hour, bursts_per_hour):
    busy = req_per_hour * BUSY_S
    up = min(3600, busy + bursts_per_hour * (COLD_START_S + IDLE_WINDOW_S))
    return up * SERVERLESS_PER_SECOND, up


if __name__ == "__main__":
    print(f"dedicated: ${DEDICATED_PER_HOUR:.2f}/h always.  serverless: ${SERVERLESS_PER_SECOND*3600:.2f}/h while up, "
          f"{COLD_START_S}s cold start, {IDLE_WINDOW_S}s idle window\n")
    print(f"{'requests/h':>11} {'bursts/h':>9} {'GPU busy':>9} {'serverless up':>14} {'serverless $/h':>15} {'cheaper':>10}")
    for rph, bursts in ((10, 2), (60, 4), (200, 6), (400, 6), (500, 8), (800, 4)):
        cost, up = serverless_cost_per_hour(rph, bursts)
        busy = rph * BUSY_S / 3600
        print(f"{rph:>11} {bursts:>9} {busy:>8.0%} {up/3600:>13.0%} {cost:>15.2f} {'serverless' if cost < DEDICATED_PER_HOUR else 'dedicated':>10}")

    # crossover with steady traffic (one burst per hour, container effectively always warm once busy)
    rph = next(r for r in range(1, 5000) if serverless_cost_per_hour(r, 1)[0] >= DEDICATED_PER_HOUR)
    print(f"\nsteady traffic crossover: ~{rph} requests/h (GPU {rph*BUSY_S/3600:.0%} busy) -> above this, dedicated is cheaper")
    print(f"first request after idle waits ~{COLD_START_S + BUSY_S:.0f} s instead of ~{BUSY_S:.0f} s; "
          "after every quiet gap longer than the idle window")
