"""Fused temperature sampling in Triton with the Gumbel-max trick.

Sampling token i with probability softmax(logits / T)_i is the same as taking
argmax_i(logits_i / T + g_i), where each g_i = -log(-log(u_i)) and u_i is uniform in (0, 1).
So one pass over the logits, with random numbers made inside the kernel, replaces
softmax + cumulative sum + search, and never writes probabilities to memory.

One program per row (request). It walks the vocabulary in blocks and keeps a running
maximum and its index. Temperature 0 means greedy (no noise). Each row has its own seed,
so a request's samples are reproducible whatever else is in the batch.

Runs on a GPU, or on a CPU with TRITON_INTERPRET=1 set before importing triton.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gumbel_sample_kernel(logits_ptr, temp_ptr, seed_ptr, out_ptr, stride, V,
                          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    temp = tl.load(temp_ptr + row)
    seed = tl.load(seed_ptr + row)
    greedy = temp == 0.0
    inv_t = tl.where(greedy, 1.0, 1.0 / tl.where(greedy, 1.0, temp))

    best_val = -float("inf")
    best_idx = 0
    for v0 in range(0, V, BLOCK):
        cols = v0 + tl.arange(0, BLOCK)
        mask = cols < V
        x = tl.load(logits_ptr + row * stride + cols, mask=mask, other=-float("inf")).to(tl.float32)
        u = tl.rand(seed, cols)                                      # counter-based: same seed, same numbers
        u = tl.minimum(tl.maximum(u, 1e-10), 1.0 - 1e-7)             # keep log(-log(u)) finite
        g = -tl.log(-tl.log(u))
        score = tl.where(greedy, x, x * inv_t + g)
        score = tl.where(mask, score, -float("inf"))
        blk_val = tl.max(score, axis=0)
        blk_idx = tl.argmax(score, axis=0) + v0
        take = blk_val > best_val
        best_idx = tl.where(take, blk_idx, best_idx)
        best_val = tl.where(take, blk_val, best_val)
    tl.store(out_ptr + row, best_idx)


def sample(logits, temperature, seeds, block=1024):
    """logits: [B, V]; temperature: [B] float32 (0 = greedy); seeds: [B] int32. Returns [B] int64."""
    B, V = logits.shape
    logits = logits.contiguous()
    out = torch.empty(B, dtype=torch.int64, device=logits.device)
    _gumbel_sample_kernel[(B,)](logits, temperature, seeds, out, logits.stride(0), V, BLOCK=block)
    return out


def sample_reference(logits, temperature, generator=None):
    """The usual unfused way: softmax, then a multinomial draw (greedy where T == 0)."""
    out = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
    greedy = temperature == 0
    out[greedy] = logits[greedy].argmax(dim=-1)
    if (~greedy).any():
        probs = torch.softmax(logits[~greedy].float() / temperature[~greedy, None], dim=-1)
        out[~greedy] = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    return out
