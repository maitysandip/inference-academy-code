"""Predict and report how one model scales across 1, 2, 4 and 8 GPUs (standard library only).

  python3 scaling_report.py predict                    # tensor-parallel decode model, defaults below
  python3 scaling_report.py predict --batch 64 --link-gbs 150
  python3 scaling_report.py report tp1:1:tp1.json tp2:2:tp2.json tp4:4:tp4.json tp8:8:tp8.json

predict: a first-order model of one decode step with tensor parallelism. Each GPU reads 1/TP
of the weights and does 1/TP of the FLOPs (the step takes whichever is longer), and each layer does two all-reduces of the batch's
hidden states (after attention and after the MLP). An all-reduce of S bytes over a ring of
TP GPUs moves 2(TP - 1)/TP x S bytes per GPU, plus a fixed latency per call.
Every default is an assumption: replace the bandwidths and latency with your own
measurements (nvidia-smi, all_reduce_perf from nccl-tests).

report: reads the JSON files written by `vllm bench serve --save-result` for each
configuration (label:gpus:file) and prints throughput, throughput per GPU, scaling
efficiency against the first configuration, and latency percentiles.
"""
import argparse
import json
import sys


def decode_step(tp, params_b, bytes_per_param, layers, hidden, batch,
                hbm_gbs, hbm_eff, link_gbs, allreduce_us, tflops=989, mfu=0.5):
    """Returns (step seconds, max(memory, compute) seconds, communication seconds).

    Ignores KV cache reads and kernel launch overhead (use CUDA graphs when you measure)."""
    weight_bytes = params_b * 1e9 * bytes_per_param / tp
    t_mem = weight_bytes / (hbm_gbs * 1e9 * hbm_eff)
    t_compute = 2 * params_b * 1e9 * batch / tp / (tflops * 1e12 * mfu)
    t_mem = max(t_mem, t_compute)
    if tp == 1:
        return t_mem, t_mem, 0.0
    msg = batch * hidden * 2                                   # activations in 16-bit
    per_call = 2 * (tp - 1) / tp * msg / (link_gbs * 1e9) + allreduce_us * 1e-6
    t_comm = 2 * layers * per_call                             # two all-reduces per layer
    return t_mem + t_comm, t_mem, t_comm


def predict(a):
    print(f"model {a.params_b}B params x {a.bytes_per_param} B, {a.layers} layers, hidden {a.hidden}, batch {a.batch}")
    print(f"HBM {a.hbm_gbs} GB/s x {a.hbm_eff}, {a.tflops} TFLOP/s x {a.mfu}, all-reduce bus bandwidth {a.link_gbs} GB/s, "
          f"{a.allreduce_us} us per all-reduce (assumptions: measure your own)\n")
    print(f"{'TP':>3} {'step ms':>9} {'mem/comp ms':>11} {'comm ms':>9} {'tok/s':>9} {'speed-up':>9} {'efficiency':>11}")
    base = None
    for tp in (1, 2, 4, 8):
        t, tm, tc = decode_step(tp, a.params_b, a.bytes_per_param, a.layers, a.hidden, a.batch,
                                a.hbm_gbs, a.hbm_eff, a.link_gbs, a.allreduce_us, a.tflops, a.mfu)
        tok = a.batch / t
        base = base or tok
        print(f"{tp:>3} {t * 1e3:9.2f} {tm * 1e3:11.2f} {tc * 1e3:9.2f} {tok:9.0f} {tok / base:8.2f}x {tok / base / tp:10.0%}")


def load(path):
    with open(path) as f:
        return json.load(f)


def report(specs):
    rows = []
    for spec in specs:
        label, gpus, path = spec.split(":", 2)
        r = load(path)
        rows.append((label, int(gpus), r))
    base_per_gpu = rows[0][2]["output_throughput"] / rows[0][1]
    hdr = f"{'config':10} {'GPUs':>4} {'out tok/s':>10} {'per GPU':>9} {'efficiency':>10} {'TTFT p50':>9} {'TTFT p99':>9} {'TPOT p50':>9} {'TPOT p99':>9} {'goodput':>8}"
    print(hdr)
    for label, gpus, r in rows:
        per_gpu = r["output_throughput"] / gpus
        good = r.get("request_goodput")
        print(f"{label:10} {gpus:>4} {r['output_throughput']:10.0f} {per_gpu:9.0f} {per_gpu / base_per_gpu:10.0%} "
              f"{r.get('median_ttft_ms', float('nan')):9.1f} {r.get('p99_ttft_ms', float('nan')):9.1f} "
              f"{r.get('median_tpot_ms', float('nan')):9.1f} {r.get('p99_tpot_ms', float('nan')):9.1f} "
              f"{'' if good is None else f'{good:8.2f}'}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    # Defaults: a 14B dense model (for example Qwen2.5-14B: 48 layers, hidden 5120) in bf16 on
    # H100 SXM-class GPUs. Check your model's config.json and your GPUs' datasheet.
    p.add_argument("--params-b", type=float, default=14.7)
    p.add_argument("--bytes-per-param", type=float, default=2)
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--hbm-gbs", type=float, default=3350)
    p.add_argument("--hbm-eff", type=float, default=0.8)
    p.add_argument("--link-gbs", type=float, default=400, help="measured all-reduce bus bandwidth")
    p.add_argument("--allreduce-us", type=float, default=10, help="fixed latency per all-reduce call")
    p.add_argument("--tflops", type=float, default=989, help="dense 16-bit tensor TFLOP/s per GPU")
    p.add_argument("--mfu", type=float, default=0.5, help="share of that peak a decode step reaches")
    r = sub.add_parser("report")
    r.add_argument("specs", nargs="+", help="label:gpus:results.json")
    a = ap.parse_args(argv)
    if a.cmd == "predict":
        predict(a)
    else:
        report(a.specs)


if __name__ == "__main__":
    main(sys.argv[1:])
