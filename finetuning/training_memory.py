GB = 1e9
shapes = {  # params (B), hidden, layers, kv_dim, inter, vocab
    "1B (Llama-3.2-1B shape)": (1.24, 2048, 16, 512, 8192),
    "8B (Llama-3.1-8B shape)": (8.03, 4096, 32, 1024, 14336),
    "70B (Llama-3.1-70B shape)": (70.55, 8192, 80, 1024, 28672),
}
def lora_params(h, L, kv, inter, r):
    return L * r * (2*(h+h) + 2*(h+kv) + 3*(h+inter))      # q,o; k,v; gate,up,down
def act_bytes(h, L, batch, seq, ckpt):
    per_layer_full = 34 * seq * batch * h                   # bytes per layer, 16-bit, with fused attention (Korthikanti et al. style estimate)
    if ckpt: return L * 2 * seq * batch * h + per_layer_full      # layer inputs kept, one layer recomputed at a time
    return L * per_layer_full
def estimate(name, method, batch=4, seq=2048, r=16, ckpt=True):
    P, h, L, kv, inter = shapes[name]; P *= 1e9
    A = lora_params(h, L, kv, inter, r)
    if method == "full":      static = P*16                 # bf16 weights 2 + bf16 grads 2 + fp32 master 4 + Adam m,v 8
    elif method == "lora":    static = P*2 + A*16           # frozen bf16 base + fp32 adapter, grads, Adam states
    elif method == "qlora":   static = P*(4.127/8) + A*16  # NF4 + double quant, adapters as above
    return static, act_bytes(h, L, batch, seq, ckpt), A
print(f"{'model':<28}{'method':<8}{'weights+states':>16}{'activations':>13}{'total':>9}   adapter params")
for name in shapes:
    for m in ("full", "lora", "qlora"):
        s, a, A = estimate(name, m)
        print(f"{name:<28}{m:<8}{s/GB:>13.1f} GB{a/GB:>10.1f} GB{(s+a)/GB*1.0:>7.0f} GB   {A/1e6 if m!='full' else 0:,.0f}M")
    print()
print("activations for 8B, batch 4, seq 2048:")
for ck in (False, True):
    print(f"  gradient checkpointing {'on ' if ck else 'off'}: {act_bytes(4096,32,4,2048,ck)/GB:5.1f} GB")
print("activations for 8B with checkpointing, by sequence length (batch 4):")
for sq in (512, 1024, 2048, 4096, 8192):
    print(f"  seq {sq:>5}: {act_bytes(4096,32,4,sq,True)/GB:5.1f} GB")
print("QLoRA 8B total by batch size at seq 2048, checkpointing on:")
for b in (1,2,4,8,16):
    s,a,_ = estimate("8B (Llama-3.1-8B shape)", "qlora", batch=b); print(f"  batch {b:>2}: {(s+a)/GB:5.1f} GB")
print("QLoRA 8B rank effect (batch 4):")
for r in (8,16,64,128):
    s,a,A = estimate("8B (Llama-3.1-8B shape)", "qlora", r=r); print(f"  r {r:>3}: adapters {A/1e6:6.1f}M -> total {(s+a)/GB:5.1f} GB")
