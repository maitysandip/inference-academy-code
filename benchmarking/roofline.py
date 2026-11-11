"""Roofline estimates for decode and prefill (datasheet figures are approximate: check current specs)."""
GPUS = {"A100-80GB": (2.0e12, 312e12), "H100-SXM": (3.35e12, 989e12), "RTX4090": (1.008e12, 165e12)}

def step_time(params, bytes_per_param, batch, bw, flops, kv_bytes_per_seq=0):
    mem = (params * bytes_per_param + batch * kv_bytes_per_seq) / bw
    comp = 2 * params * batch / flops
    return max(mem, comp), mem, comp

if __name__ == "__main__":
    for name, (bw, fl) in GPUS.items():
        print(name, "ridge", round(fl / bw, 1))
        for b in (1, 8, 64, 256, 512):
            t, m, c = step_time(8e9, 2, b, bw, fl)
            print("  batch", b, f"{t*1e3:.2f} ms", f"{b/t:,.0f} tok/s", "memory" if m >= c else "compute")
