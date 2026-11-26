"""One benchmark harness for the kernel pack: PyTorch eager, torch.compile, Triton and CUDA.

For each operation it builds realistic inputs, checks every backend against eager PyTorch,
then (on a GPU) times each one with triton.testing.do_bench (median of runs, L2 cache cleared
before each) and reports the time, the
achieved memory bandwidth or FLOP rate, and the speed-up over eager.

  python3 bench.py                      # every op, on a GPU
  python3 bench.py --ops rope,layernorm
  python3 bench.py --check-only         # correctness only (works on a CPU with TRITON_INTERPRET=1)

The CUDA backend exists for rope and layernorm (cuda_kernels.cu, built on first use with
torch.utils.cpp_extension, which needs nvcc). Missing backends are reported, not fatal.
"""
import argparse
import os
import sys

import torch

GPU = torch.cuda.is_available()
if not GPU:
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layernorm as ln  # noqa: E402
import moe as moe_mod  # noqa: E402
import rope as rope_mod  # noqa: E402
import sampling as samp  # noqa: E402
import w4a16_matmul as w4  # noqa: E402

DEV = "cuda" if GPU else "cpu"
_cuda_ext = None


def cuda_ext():
    """Build cuda_kernels.cu once. Raises if there is no GPU or no nvcc."""
    global _cuda_ext
    if _cuda_ext is None:
        if not GPU:
            raise RuntimeError("no GPU")
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _cuda_ext = load("ia_cuda_kernels", [os.path.join(here, "cuda_kernels.cu")], verbose=False)
    return _cuda_ext


# ---------------------------------------------------------------- the operations
# Each op: make(small) -> inputs, eager(*inputs), triton(*inputs), optional cuda(*inputs),
# cost(*inputs) -> (bytes moved, flops), and a tolerance for the comparison.

def rope_eager(q, k, pos, base=10000.0):
    """The usual PyTorch version: build cos and sin, rotate half, in the input dtype."""
    d = q.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, device=q.device, dtype=torch.float32) / d)
    ang = pos.float()[:, None] * inv[None]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None].to(q.dtype)
    sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None].to(q.dtype)

    def rot(x):
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin
    return rot(q), rot(k)


OPS = {
    "rope": dict(
        make=lambda small: (torch.randn(64 if small else 4096, 8 if small else 32, 128, device=DEV,
                                        dtype=torch.float32 if small else torch.float16),
                            torch.randn(64 if small else 4096, 2 if small else 8, 128, device=DEV,
                                        dtype=torch.float32 if small else torch.float16),
                            torch.arange(64 if small else 4096, device=DEV)),
        eager=rope_eager,
        triton=lambda q, k, p: rope_mod.rope_(q.clone(), k.clone(), p),
        cuda=lambda q, k, p: (lambda q2, k2: (cuda_ext().rope_(q2, k2, p, 10000.0), (q2, k2))[1])(q.clone(), k.clone()),
        cost=lambda q, k, p: (2 * (q.numel() + k.numel()) * q.element_size(), 0),
        tol=2e-2,
    ),
    "layernorm": dict(
        make=lambda small: (torch.randn(32 if small else 4096, 1000 if small else 4096, device=DEV,
                                        dtype=torch.float32 if small else torch.float16),
                            torch.randn(1000 if small else 4096, device=DEV, dtype=torch.float32 if small else torch.float16),
                            torch.randn(1000 if small else 4096, device=DEV, dtype=torch.float32 if small else torch.float16)),
        eager=lambda x, w, b: torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b),
        triton=ln.layernorm,
        cuda=lambda x, w, b: cuda_ext().layernorm(x, w, b, 1e-5),
        cost=lambda x, w, b: (2 * x.numel() * x.element_size(), 0),
        tol=2e-2,
    ),
    "w4a16": dict(
        make=lambda small: (lambda K, N, M: (torch.randn(M, K, device=DEV).half(),
                                             *[t.to(DEV) for t in w4.quantise_w4(torch.randn(K, N), 128)]))(
            256 if small else 4096, 192 if small else 11008, 4 if small else 16),
        eager=lambda x, p, s: (x.float() @ w4.dequantise_w4(p, s)).half() if not GPU
        else x @ w4.dequantise_w4(p, s).half(),              # dequantise the whole weight, then matmul
        triton=w4.w4a16_matmul,
        cost=lambda x, p, s: (p.numel() + s.numel() * 2 + x.numel() * 2 + x.shape[0] * p.shape[1] * 4,
                              2 * x.shape[0] * x.shape[1] * p.shape[1] * 2),
        tol=5e-1,
    ),
    "moe": dict(
        make=lambda small: (lambda T, K, N, E: (torch.randn(T, K, device=DEV, dtype=torch.float32 if small else torch.float16),
                                                torch.randn(E, K, N, device=DEV, dtype=torch.float32 if small else torch.float16) * 0.05,
                                                torch.randn(T, E, device=DEV)))(
            37 if small else 256, 64 if small else 2048, 96 if small else 1536, 8 if small else 64),
        eager=lambda x, w, g: moe_mod.moe_reference(x, w, g, 2 if x.shape[0] < 100 else 6),
        triton=lambda x, w, g: moe_mod.moe_linear(x, w, g, 2 if x.shape[0] < 100 else 6),
        cost=lambda x, w, g: (w.numel() * w.element_size() + 2 * x.numel() * x.element_size(),
                              2 * x.shape[0] * (2 if x.shape[0] < 100 else 6) * w.shape[1] * w.shape[2]),
        tol=5e-2,
    ),
    "sampling": dict(
        make=lambda small: (torch.randn(4 if small else 64, 3000 if small else 128256, device=DEV),
                            torch.zeros(4 if small else 64, device=DEV),          # greedy: comparable to eager exactly
                            torch.arange(4 if small else 64, dtype=torch.int32, device=DEV)),
        eager=lambda l, t, s: samp.sample_reference(l, t),
        triton=samp.sample,
        cost=lambda l, t, s: (l.numel() * l.element_size(), 0),
        tol=0,
    ),
}


def first(x):
    return x[0] if isinstance(x, tuple) else x


def check(name, op, small):
    torch.manual_seed(0)
    inputs = op["make"](small)
    ref = first(op["eager"](*inputs)).float()
    results = {}
    backends = {"compile": torch.compile(op["eager"]), "triton": op["triton"]}
    if "cuda" in op:
        backends["cuda"] = op["cuda"]
    for b, fn in backends.items():
        try:
            out = first(fn(*inputs)).float()
            err = (out - ref).abs().max().item()
            results[b] = ("ok" if err <= op["tol"] else "MISMATCH", err)
        except Exception as e:                                 # missing compiler, no GPU, no nvcc...
            results[b] = ("unavailable", f"{type(e).__name__}: {str(e).splitlines()[0][:60]}")
    return inputs, backends, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default=",".join(OPS))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    check_only = args.check_only or not GPU
    if not GPU:
        print("No GPU: checking correctness with small inputs in the Triton interpreter; no timings.\n")
    else:
        print(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}, triton {triton.__version__}\n")

    print(f"{'op':10} {'backend':8} {'check':12} {'max err':>9} {'time us':>9} {'GB/s':>8} {'TFLOP/s':>8} {'vs eager':>8}")
    for name in args.ops.split(","):
        op = OPS[name]
        inputs, backends, results = check(name, op, small=check_only)
        rows = [("eager", op["eager"], ("ref", 0.0))] + [(b, fn, results[b]) for b, fn in backends.items()]
        t_eager = None
        for b, fn, (status, err) in rows:
            t = gbs = tf = speed = ""
            if not check_only and status in ("ok", "ref"):
                ms = triton.testing.do_bench(lambda: fn(*inputs), warmup=25, rep=100, return_mode="median")
                nbytes, flops = op["cost"](*inputs)
                t_eager = t_eager or ms
                t, gbs = f"{ms * 1e3:9.1f}", f"{nbytes / ms / 1e6:8.0f}"
                tf = f"{flops / ms / 1e9:8.1f}" if flops else ""
                speed = f"{t_eager / ms:7.2f}x"
            err_s = f"{err:9.2e}" if isinstance(err, float) else ""
            print(f"{name:10} {b:8} {status:12} {err_s:>9} {t:>9} {gbs:>8} {tf:>8} {speed:>8}"
                  + (f"  {err}" if status == "unavailable" else ""))
        print()


if __name__ == "__main__":
    main()
