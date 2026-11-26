"""LayerNorm forward in Triton: one program per row, statistics in float32.

y = (x - mean) / sqrt(var + eps) * weight + bias, over the last dimension.
For rows that fit in one block (hidden sizes up to a few thousand) the whole row stays in
registers, so x is read once and y written once: the kernel is bound by memory bandwidth.

Runs on a GPU, or on a CPU with TRITON_INTERPRET=1 set before importing triton.
"""
import torch
import triton
import triton.language as tl


def layernorm_reference(x, weight, bias, eps=1e-5):
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).to(x.dtype)


@triton.jit
def _layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, stride, n_cols, eps,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / n_cols
    centred = tl.where(mask, x - mean, 0.0)          # padding must not count towards the variance
    var = tl.sum(centred * centred, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = centred * rstd * w + b
    tl.store(y_ptr + row * stride + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def layernorm(x, weight, bias, eps=1e-5):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    y = torch.empty_like(x2)
    n_rows, n_cols = x2.shape
    block = triton.next_power_of_2(n_cols)
    assert block <= 16384, "rows this long need a looped kernel (see the exercise)"
    _layernorm_kernel[(n_rows,)](x2, weight, bias, y, x2.stride(0), n_cols, eps, BLOCK=block)
    return y.reshape(shape)
