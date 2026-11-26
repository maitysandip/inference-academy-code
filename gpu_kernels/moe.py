"""Mixture-of-experts dispatch and a grouped GEMM in Triton.

Steps, for T tokens, E experts and top_k experts per token:
  1. route:    softmax over router logits, keep the top_k, renormalise their weights.
  2. align:    sort the T * top_k (token, expert) assignments by expert and pad each expert's
               list to a multiple of BLOCK_M, so every block of rows belongs to one expert.
  3. grouped GEMM: one program per (row block, column block). It reads which expert its block
               belongs to, gathers those tokens' rows of x, multiplies by that expert's weight
               and scales by the routing weight (fused), writing one output row per assignment.
  4. combine:  sum each token's top_k rows.

A real expert is an MLP (two grouped GEMMs with an activation between); one linear layer per
expert keeps the dispatch logic in view. Runs on a GPU, or on a CPU with TRITON_INTERPRET=1.
"""
import torch
import triton
import triton.language as tl


def route(router_logits, top_k):
    probs = torch.softmax(router_logits.float(), dim=-1)
    weights, ids = probs.topk(top_k, dim=-1)                 # [T, top_k]
    return weights / weights.sum(dim=-1, keepdim=True), ids


def align_by_expert(topk_ids, num_experts, block_m):
    """Returns sorted_ids (padded with the sentinel T*top_k) and one expert id per block of rows."""
    flat = topk_ids.reshape(-1)
    sentinel = flat.numel()
    order = torch.argsort(flat, stable=True)                 # assignment indices grouped by expert
    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block_m - 1) // block_m * block_m
    sorted_ids = torch.full((int(padded.sum()),), sentinel, dtype=torch.int32, device=flat.device)
    expert_of_block = torch.repeat_interleave(torch.arange(num_experts, device=flat.device), padded // block_m)
    starts = torch.cumsum(counts, 0) - counts
    dst = torch.cumsum(padded, 0) - padded
    for e in range(num_experts):                             # small loop over experts; real engines use a kernel
        c = int(counts[e])
        sorted_ids[int(dst[e]): int(dst[e]) + c] = order[int(starts[e]): int(starts[e]) + c].to(torch.int32)
    return sorted_ids, expert_of_block.to(torch.int32)


@triton.jit
def _grouped_gemm_kernel(x_ptr, w_ptr, out_ptr, sorted_ids_ptr, expert_ptr, weight_ptr,
                         num_assign, N, K, top_k,
                         stride_xt, stride_we, stride_wk, stride_out,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.load(expert_ptr + pid_m)

    rows = tl.load(sorted_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))   # assignment ids
    valid = rows < num_assign                                                  # padding rows are skipped
    token = rows // top_k                                                      # assignment -> token
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        a = tl.load(x_ptr + token[:, None] * stride_xt + kk[None, :],
                    mask=valid[:, None] & (kk[None, :] < K), other=0.0)
        b = tl.load(w_ptr + expert * stride_we + kk[:, None] * stride_wk + rn[None, :],
                    mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    gate = tl.load(weight_ptr + rows, mask=valid, other=0.0)                   # fused routing weight
    acc = acc * gate[:, None]
    tl.store(out_ptr + rows[:, None] * stride_out + rn[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=valid[:, None] & (rn[None, :] < N))


def moe_linear(x, w, router_logits, top_k, block_m=16, block_n=64, block_k=32):
    """x: [T, K], w: [E, K, N], router_logits: [T, E]. Returns [T, N]."""
    T, K = x.shape
    E, _, N = w.shape
    weights, ids = route(router_logits, top_k)
    sorted_ids, expert_of_block = align_by_expert(ids, E, block_m)
    out = torch.zeros((T * top_k, N), dtype=x.dtype, device=x.device)
    flat_weights = weights.reshape(-1).to(x.dtype).contiguous()
    grid = (expert_of_block.numel(), triton.cdiv(N, block_n))
    _grouped_gemm_kernel[grid](x, w, out, sorted_ids, expert_of_block, flat_weights,
                               T * top_k, N, K, top_k,
                               x.stride(0), w.stride(0), w.stride(1), out.stride(0),
                               BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return out.view(T, top_k, N).sum(dim=1)


def moe_reference(x, w, router_logits, top_k):
    weights, ids = route(router_logits, top_k)
    out = torch.zeros((x.shape[0], w.shape[2]), dtype=torch.float32, device=x.device)
    for e in range(w.shape[0]):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel():
            out[tok] += weights[tok, slot][:, None] * (x[tok].float() @ w[e].float())
    return out.to(x.dtype)
