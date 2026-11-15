import math, random
from statistics import NormalDist
random.seed(3)

# 1. LoRA parameter counts for a Llama-3.1-8B-shaped model
hidden, layers, kv_dim, inter = 4096, 32, 1024, 14336
mats = {"q_proj": (hidden, hidden), "k_proj": (hidden, kv_dim), "v_proj": (hidden, kv_dim), "o_proj": (hidden, hidden),
        "gate_proj": (hidden, inter), "up_proj": (hidden, inter), "down_proj": (inter, hidden)}
base = layers * sum(i * o for i, o in mats.values()) + 2 * 128256 * hidden
print(f"base ≈ {base/1e9:.2f}B params")
def lora_params(r, targets): return layers * sum(r * (mats[t][0] + mats[t][1]) for t in targets)
attn = ["q_proj", "k_proj", "v_proj", "o_proj"]; allm = list(mats)
print(f"{'rank':>5}  {'attention only':>22}  {'all linear layers':>22}")
for r in (4, 8, 16, 64, 128):
    a, b = lora_params(r, attn), lora_params(r, allm)
    print(f"{r:>5}  {a/1e6:8.1f}M ({a/base:6.3%})  {b/1e6:9.1f}M ({b/base:6.3%})   adapter file ≈ {b*2/1e6:6.0f} MB in BF16")

# 2. LoRA forward pass equals the merged weights
def matvec(W, x): return [sum(w*v for w, v in zip(row, x)) for row in W]
def matmul(A, B): return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
d, r, alpha = 6, 2, 4
W = [[random.gauss(0, 1) for _ in range(d)] for _ in range(d)]
A = [[random.gauss(0, 0.1) for _ in range(d)] for _ in range(r)]      # r x d, random
B = [[random.gauss(0, 0.1) for _ in range(r)] for _ in range(d)]      # d x r (after training; starts at zero)
x = [random.gauss(0, 1) for _ in range(d)]
scale = alpha / r
adapter_out = [w + scale * u for w, u in zip(matvec(W, x), matvec(B, matvec(A, x)))]
BA = matmul(B, A)
merged = [[W[i][j] + scale * BA[i][j] for j in range(d)] for i in range(d)]
merged_out = matvec(merged, x)
print("\nadapter path vs merged weights, max difference:", f"{max(abs(a-b) for a, b in zip(adapter_out, merged_out)):.2e}")
print("at initialisation (B = 0) the update is exactly:", matvec([[0.0]*r for _ in range(d)], matvec(A, x)))

# 3. How much does rank matter? Approximate a "fine-tuning update" of known intrinsic rank
n, true_rank = 40, 4
U = [[random.gauss(0, 1) for _ in range(true_rank)] for _ in range(n)]
V = [[random.gauss(0, 1) for _ in range(n)] for _ in range(true_rank)]
delta = matmul(U, V)
noise = [[random.gauss(0, 0.3) for _ in range(n)] for _ in range(n)]
dW = [[delta[i][j] + noise[i][j] for j in range(n)] for i in range(n)]
def fro(M): return math.sqrt(sum(v*v for row in M for v in row))
def top_component(M, iters=60):
    n_r, n_c = len(M), len(M[0])
    v = [random.gauss(0, 1) for _ in range(n_c)]
    for _ in range(iters):
        u = matvec(M, v); nu = math.sqrt(sum(t*t for t in u)); u = [t/nu for t in u]
        v = [sum(M[i][j]*u[i] for i in range(n_r)) for j in range(n_c)]; nv = math.sqrt(sum(t*t for t in v)); v = [t/nv for t in v]
    sigma = sum(u[i]*sum(M[i][j]*v[j] for j in range(n_c)) for i in range(n_r))
    return sigma, u, v
resid = [row[:] for row in dW]
print("\nrank   energy captured   relative error")
for k in range(1, 9):
    s, u, v = top_component(resid)
    resid = [[resid[i][j] - s*u[i]*v[j] for j in range(n)] for i in range(n)]
    print(f"{k:>4}   {1 - (fro(resid)/fro(dW))**2:14.1%}   {fro(resid)/fro(dW):14.3f}")

# 4. Why 4-bit "normal float" beats uniform 4-bit on bell-shaped weights
nd = NormalDist()
w = [random.gauss(0, 1) for _ in range(20000)]
absmax = max(abs(x) for x in w)
uniform = [(-absmax + 2*absmax*i/15) for i in range(16)]
quant_levels = sorted(nd.inv_cdf((i + 0.5)/16) for i in range(16)); m = max(abs(q) for q in quant_levels); quant_levels = [q/m*absmax for q in quant_levels]
def mse(levels): return sum(min((x-l)**2 for l in levels) for x in w)/len(w)
print(f"\n4-bit uniform levels   mean squared error {mse(uniform):.4f}")
print(f"4-bit quantile levels  mean squared error {mse(quant_levels):.4f}  ({mse(uniform)/mse(quant_levels):.1f}x lower)")
