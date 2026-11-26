"""Measure a noisy neighbour on a shared GPU: how much does another process slow yours down?

The victim runs a decode-like step (a small batch times a large weight matrix, the shape of
one linear layer at decode) many times and records the latency of each step. It runs once
alone, then once with an aggressor process hammering the same GPU, and reports p50, p99 and
the slowdown. Run it under each sharing mode and compare:

  time-slicing (the default when two processes share a GPU, no MPS daemon):
      python3 noisy_neighbour.py
  MPS (start the daemon first: nvidia-cuda-mps-control -d; stop: echo quit | nvidia-cuda-mps-control):
      python3 noisy_neighbour.py
  MIG (victim and aggressor on different MIG instances; list UUIDs with nvidia-smi -L):
      python3 noisy_neighbour.py --victim-device MIG-<uuid-a> --aggressor-device MIG-<uuid-b>

Aggressors: "compute" (large matmuls) or "memory" (large copies, bandwidth bound).
Needs PyTorch. --cpu runs a tiny version on the CPU to check the script itself, not a GPU.
"""
import argparse
import os
import statistics
import time

import multiprocessing as mp


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def aggressor(kind, device_env, cpu, ready, stop):
    if device_env:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_env
    import torch
    dev = "cpu" if cpu else "cuda"
    n = 256 if cpu else 8192
    if kind == "compute":
        a = torch.randn(n, n, device=dev, dtype=torch.float32 if cpu else torch.float16)
        b = torch.randn_like(a)
        work = lambda: a @ b                                   # noqa: E731
    else:
        src = torch.empty(64 * 2**20 if not cpu else 2**20, dtype=torch.uint8, device=dev)
        dst = torch.empty_like(src)
        work = lambda: dst.copy_(src)                          # noqa: E731
    work()
    ready.set()
    while not stop.is_set():
        for _ in range(20):
            work()
        if not cpu:
            torch.cuda.synchronize()


def victim_steps(steps, cpu, batch, hidden):
    import torch
    dev = "cpu" if cpu else "cuda"
    dtype = torch.float32 if cpu else torch.float16
    x = torch.randn(batch, hidden, device=dev, dtype=dtype)
    w = torch.randn(hidden, hidden, device=dev, dtype=dtype)
    for _ in range(10):                                        # warm-up
        x @ w
    lat = []
    if cpu:
        for _ in range(steps):
            t0 = time.perf_counter(); x @ w; lat.append((time.perf_counter() - t0) * 1e3)
        return lat
    start = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    for i in range(steps):
        start[i].record(); x @ w; end[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(start, end)]


def report(name, lat, base=None):
    p50, p99 = percentile(lat, 50), percentile(lat, 99)
    extra = f"   slowdown p50 {p50 / base[0]:.2f}x, p99 {p99 / base[1]:.2f}x" if base else ""
    print(f"{name:28s} p50 {p50:8.3f} ms   p99 {p99:8.3f} ms   mean {statistics.mean(lat):8.3f} ms{extra}")
    return p50, p99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=8192)
    ap.add_argument("--aggressors", default="compute,memory")
    ap.add_argument("--victim-device", default="", help="CUDA_VISIBLE_DEVICES for the victim (e.g. a MIG UUID)")
    ap.add_argument("--aggressor-device", default="", help="CUDA_VISIBLE_DEVICES for the aggressor")
    ap.add_argument("--cpu", action="store_true", help="tiny CPU run to check the script, not a GPU measurement")
    args = ap.parse_args()
    if args.cpu:
        args.steps, args.hidden = min(args.steps, 200), 256
    if args.victim_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.victim_device

    import torch
    if not args.cpu:
        print(f"victim GPU: {torch.cuda.get_device_name()}  (MPS daemon running: {os.path.exists('/tmp/nvidia-mps')})")
    print(f"victim step: [{args.batch} x {args.hidden}] @ [{args.hidden} x {args.hidden}], {args.steps} steps\n")

    base = report("alone", victim_steps(args.steps, args.cpu, args.batch, args.hidden))
    ctx = mp.get_context("spawn")
    for kind in args.aggressors.split(","):
        ready, stop = ctx.Event(), ctx.Event()
        p = ctx.Process(target=aggressor, args=(kind, args.aggressor_device, args.cpu, ready, stop))
        p.start()
        ready.wait(120)
        lat = victim_steps(args.steps, args.cpu, args.batch, args.hidden)
        stop.set(); p.join(30)
        report(f"with a {kind} aggressor", lat, base)


if __name__ == "__main__":
    main()
