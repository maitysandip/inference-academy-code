"""Full fine-tuning memory per GPU with data parallelism and ZeRO / FSDP sharding.

Mixed-precision Adam keeps 16 bytes per parameter: bf16 weights (2), bf16 gradients (2),
fp32 master weights (4), fp32 Adam moments (8). Plain data parallelism copies all of it to every GPU.
ZeRO stage 1 shards the optimizer states (the 12 fp32 bytes), stage 2 also the gradients,
stage 3 (and FSDP full sharding) also the weights. Activations are not sharded by ZeRO and are
listed separately (8B shape, batch 1 per GPU, seq 4096, checkpointing on, rough estimate).
"""
GB = 1e9


def per_gpu(params, gpus, stage):
    w, g, opt = 2 * params, 2 * params, 12 * params
    if stage >= 1:
        opt /= gpus
    if stage >= 2:
        g /= gpus
    if stage >= 3:
        w /= gpus
    return w + g + opt


def activations(h, layers, seq, batch=1):
    return layers * 2 * seq * batch * h + 34 * seq * batch * h


if __name__ == "__main__":
    models = {"8B": (8.03e9, 4096, 32), "32B": (32.8e9, 5120, 64), "70B": (70.6e9, 8192, 80)}
    for name, (P, h, L) in models.items():
        act = activations(h, L, 4096) / GB
        print(f"{name}: weights+grads+optimizer = {per_gpu(P, 1, 0)/GB:.0f} GB unsharded; activations ~{act:.1f} GB per GPU")
        print(f"  {'GPUs':>5} {'DP (no shard)':>14} {'ZeRO-1':>8} {'ZeRO-2':>8} {'ZeRO-3/FSDP':>12}")
        for n in (1, 2, 4, 8, 16):
            row = [per_gpu(P, n, s) / GB + act for s in (0, 1, 2, 3)]
            print(f"  {n:>5} {row[0]:>11.0f} GB {row[1]:>5.0f} GB {row[2]:>5.0f} GB {row[3]:>9.0f} GB")
        print()
    print("fits on 80 GB GPUs (75 GB usable) with ZeRO-3, smallest power-of-two count:")
    for name, (P, h, L) in models.items():
        act = activations(h, L, 4096) / GB
        n = next(n for n in (1, 2, 4, 8, 16, 32, 64) if per_gpu(P, n, 3) / GB + act <= 75)
        print(f"  {name}: {n} GPUs")
