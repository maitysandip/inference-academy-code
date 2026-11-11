"""Static versus continuous batching simulation."""
import random
def simulate(requests, max_batch, continuous):
    """requests: list of (arrival_step, out_tokens). One decode step = one time unit; returns mean latency and tokens/step."""
    pending = sorted(requests); t = 0; running = []; done = []; total_tokens = 0
    while pending or running:
        if continuous or not running:                      # static batching only admits when the batch is empty
            while pending and pending[0][0] <= t and len(running) < max_batch:
                a, n = pending.pop(0); running.append([a, n])
        if not running: t += 1; continue
        t += 1
        for r in running: r[1] -= 1; total_tokens += 1
        for r in [r for r in running if r[1] == 0]: done.append(t - r[0])
        running = [r for r in running if r[1] > 0]
    return sum(done) / len(done), total_tokens / t

if __name__ == "__main__":
    rng = random.Random(3)
    reqs = [(i * 2, rng.choice([5, 10, 20, 80])) for i in range(60)]
    for mode in (False, True):
        lat, thr = simulate(reqs, 8, mode); print("continuous" if mode else "static    ", "mean latency", round(lat, 1), "steps; tokens/step", round(thr, 2))
