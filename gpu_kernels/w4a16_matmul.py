"""Weight-only 4-bit matmul (W4A16) in Triton: y = x @ dequant(W).

Quantisation (symmetric, group-wise along K, as in GPTQ/AWQ-style formats without zero points):
  for each column n and each group of GROUP rows of K: scale = max|w| / 7,
  q = clamp(round(w / scale), -8, 7), stored as q + 8 in 4 bits.
Packing: two 4-bit values per byte along N. packed[k, j] holds column 2j in the low nibble
and column 2j + 1 in the high nibble, so packed is [K, N/2] uint8 and scales is [K/GROUP, N].

The kernel loads packed bytes, splits them into the even and odd columns, dequantises in
registers and runs two tensor-core dots per K step. Weights cross memory as 4 bits each,
which is the whole point for decode, where the matmul is bound by reading the weights.

Runs on a GPU, or on a CPU with TRITON_INTERPRET=1 set before importing triton.
"""
import torch
import triton
import triton.language as tl


def quantise_w4(w, group=128):
    """w: [K, N] float. Returns packed [K, N/2] uint8 and scales [K/group, N] float16."""
    K, N = w.shape
    assert K % group == 0 and N % 2 == 0
    wg = w.float().reshape(K // group, group, N)
    scales = (wg.abs().amax(dim=1) / 7.0).clamp(min=1e-8)              # [K/group, N]
    q = torch.clamp(torch.round(wg / scales[:, None, :]), -8, 7).to(torch.int16) + 8
    q = q.reshape(K, N).to(torch.uint8)
    packed = q[:, 0::2] | (q[:, 1::2] << 4)
    return packed.contiguous(), scales.to(torch.float16)


def dequantise_w4(packed, scales, group=128):
    lo = (packed & 0xF).to(torch.int16) - 8
    hi = (packed >> 4).to(torch.int16) - 8
    K = packed.shape[0]
    q = torch.stack([lo, hi], dim=-1).reshape(K, -1).float()             # [K, N], columns interleaved back
    return q * scales.float().repeat_interleave(group, dim=0)


def w4a16_reference(x, packed, scales, group=128):
    return (x.float() @ dequantise_w4(packed, scales, group)).to(x.dtype)


@triton.jit
def _w4a16_kernel(x_ptr, p_ptr, s_ptr, y_ptr, M, N, K,
                  stride_xm, stride_pk, stride_sk, stride_ym,
                  GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    HALF_N: tl.constexpr = BLOCK_N // 2

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rj = pid_n * HALF_N + tl.arange(0, HALF_N)          # packed column index: covers columns 2j, 2j+1
    rk = tl.arange(0, BLOCK_K)

    acc_even = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
    acc_odd = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * stride_xm + kk[None, :],
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        p = tl.load(p_ptr + kk[:, None] * stride_pk + rj[None, :],
                    mask=(kk[:, None] < K) & (rj[None, :] < N // 2), other=0)
        # One scale row per group; BLOCK_K divides GROUP, so the whole K block shares it.
        g = k0 // GROUP
        s_even = tl.load(s_ptr + g * stride_sk + 2 * rj, mask=rj < N // 2, other=0.0)
        s_odd = tl.load(s_ptr + g * stride_sk + 2 * rj + 1, mask=rj < N // 2, other=0.0)

        w_even = ((p & 0xF).to(tl.int32) - 8).to(tl.float16) * s_even[None, :]
        w_odd = ((p >> 4).to(tl.int32) - 8).to(tl.float16) * s_odd[None, :]
        acc_even += tl.dot(x.to(tl.float16), w_even.to(tl.float16))
        acc_odd += tl.dot(x.to(tl.float16), w_odd.to(tl.float16))

    out_dtype = y_ptr.dtype.element_ty
    mask = (rm[:, None] < M) & (rj[None, :] < N // 2)
    tl.store(y_ptr + rm[:, None] * stride_ym + 2 * rj[None, :], acc_even.to(out_dtype), mask=mask)
    tl.store(y_ptr + rm[:, None] * stride_ym + 2 * rj[None, :] + 1, acc_odd.to(out_dtype), mask=mask)


def w4a16_matmul(x, packed, scales, group=128, block_m=16, block_n=64, block_k=64):
    """x: [M, K] float16. Small block_m suits decode (M is the batch size, often 1 to 64)."""
    M, K = x.shape
    N = packed.shape[1] * 2
    assert packed.shape[0] == K and group % block_k == 0
    x = x.contiguous()
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _w4a16_kernel[grid](x, packed, scales, y, M, N, K,
                        x.stride(0), packed.stride(0), scales.stride(0), y.stride(0),
                        GROUP=group, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    return y
