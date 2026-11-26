"""Rotary position embedding (RoPE) for Q and K in one Triton kernel, in place.

Layout: q is [tokens, q_heads, head_dim], k is [tokens, kv_heads, head_dim], positions is [tokens].
Uses the "half split" (GPT-NeoX) convention: dimension i pairs with dimension i + head_dim/2.
One program handles one (token, head) pair; heads 0..q_heads-1 are Q, the rest are K.

Runs on a GPU, or on a CPU with TRITON_INTERPRET=1 set before importing triton.
"""
import torch
import triton
import triton.language as tl


def rope_reference(q, k, positions, base=10000.0):
    """Plain PyTorch, computed in float32. Returns new tensors."""
    d = q.shape[-1]
    inv_freq = base ** (-torch.arange(0, d // 2, dtype=torch.float32, device=q.device) * 2 / d)
    angles = positions.float()[:, None] * inv_freq[None, :]          # [tokens, d/2]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]    # broadcast over heads

    def rotate(x):
        x1, x2 = x.float()[..., : d // 2], x.float()[..., d // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)

    return rotate(q), rotate(k)


@triton.jit
def _rope_kernel(q_ptr, k_ptr, pos_ptr,
                 q_stride_t, q_stride_h, k_stride_t, k_stride_h,
                 n_q_heads, log_base,
                 HALF: tl.constexpr):
    t = tl.program_id(0)                     # token
    h = tl.program_id(1)                     # head: Q heads first, then K heads
    pos = tl.load(pos_ptr + t).to(tl.float32)

    i = tl.arange(0, HALF)
    inv_freq = tl.exp(-(2.0 * i.to(tl.float32) / (2 * HALF)) * log_base)
    angle = pos * inv_freq
    cos, sin = tl.cos(angle), tl.sin(angle)

    if h < n_q_heads:
        base_ptr = q_ptr + t * q_stride_t + h * q_stride_h
    else:
        base_ptr = k_ptr + t * k_stride_t + (h - n_q_heads) * k_stride_h

    x1 = tl.load(base_ptr + i).to(tl.float32)
    x2 = tl.load(base_ptr + HALF + i).to(tl.float32)
    out_dtype = base_ptr.dtype.element_ty
    tl.store(base_ptr + i, (x1 * cos - x2 * sin).to(out_dtype))
    tl.store(base_ptr + HALF + i, (x2 * cos + x1 * sin).to(out_dtype))


def rope_(q, k, positions, base=10000.0):
    """Rotate q and k in place. head_dim must be a power of two (64 and 128 are typical)."""
    tokens, n_q, d = q.shape
    n_kv = k.shape[1]
    assert k.shape[0] == tokens and k.shape[2] == d and d % 2 == 0
    assert (d // 2) & (d // 2 - 1) == 0, "head_dim / 2 must be a power of two"
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    grid = (tokens, n_q + n_kv)
    _rope_kernel[grid](q, k, positions,
                       q.stride(0), q.stride(1), k.stride(0), k.stride(1),
                       n_q, float(torch.log(torch.tensor(base))),
                       HALF=d // 2)
    return q, k
