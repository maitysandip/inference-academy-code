// Hand-written CUDA versions of two kernels from the pack, for the benchmark harness.
// Built on the fly with torch.utils.cpp_extension.load() (needs nvcc and a CUDA build of PyTorch).
// float16 inputs and outputs, float32 arithmetic, the same maths as layernorm.py and rope.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

// Sum a value across a whole thread block: shuffle within each warp, then across warps.
template <int THREADS>
__device__ float block_sum(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) red[warp] = v;
  __syncthreads();
  v = (threadIdx.x < THREADS / 32) ? red[lane] : 0.f;
  if (warp == 0)
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  if (threadIdx.x == 0) red[0] = v;
  __syncthreads();
  v = red[0];
  __syncthreads();  // red is reused by the next call
  return v;
}

// One block per row. Two passes over the row for the statistics, one for the output;
// rows are short enough (a few thousand values) to stay in L1/L2 between passes.
template <int THREADS>
__global__ void layernorm_kernel(const half* __restrict__ x, const half* __restrict__ w,
                                 const half* __restrict__ b, half* __restrict__ y, int n, float eps) {
  __shared__ float red[32];
  const half* xr = x + (size_t)blockIdx.x * n;
  half* yr = y + (size_t)blockIdx.x * n;

  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += THREADS) s += __half2float(xr[i]);
  const float mean = block_sum<THREADS>(s, red) / n;

  float v = 0.f;
  for (int i = threadIdx.x; i < n; i += THREADS) {
    const float d = __half2float(xr[i]) - mean;
    v += d * d;
  }
  const float rstd = rsqrtf(block_sum<THREADS>(v, red) / n + eps);

  for (int i = threadIdx.x; i < n; i += THREADS)
    yr[i] = __float2half((__half2float(xr[i]) - mean) * rstd * __half2float(w[i]) + __half2float(b[i]));
}

// One block per (token, head); one thread per rotated pair (i, i + half_d).
__global__ void rope_kernel(half* q, half* k, const int64_t* __restrict__ pos, int n_q, int half_d,
                            int64_t q_st, int64_t q_sh, int64_t k_st, int64_t k_sh, float log_base) {
  const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
  if (i >= half_d) return;
  half* p = (h < n_q) ? q + t * q_st + h * q_sh : k + t * k_st + (h - n_q) * k_sh;
  const float inv_freq = expf(-(2.f * i / (2 * half_d)) * log_base);
  float s, c;
  sincosf((float)pos[t] * inv_freq, &s, &c);
  const float x1 = __half2float(p[i]), x2 = __half2float(p[i + half_d]);
  p[i] = __float2half(x1 * c - x2 * s);
  p[i + half_d] = __float2half(x2 * c + x1 * s);
}

torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf && x.is_contiguous());
  auto y = torch::empty_like(x);
  const int n = x.size(-1);
  const int rows = x.numel() / n;
  layernorm_kernel<256><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()), reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b.data_ptr<at::Half>()), reinterpret_cast<half*>(y.data_ptr<at::Half>()), n,
      (float)eps);
  return y;
}

void rope_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor pos, double base) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kHalf && k.scalar_type() == torch::kHalf);
  const int half_d = q.size(2) / 2;
  TORCH_CHECK(half_d <= 1024);
  dim3 grid(q.size(0), q.size(1) + k.size(1));
  rope_kernel<<<grid, half_d, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(q.data_ptr<at::Half>()), reinterpret_cast<half*>(k.data_ptr<at::Half>()),
      pos.data_ptr<int64_t>(), q.size(1), half_d, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
      (float)std::log(base));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("layernorm", &layernorm_cuda, "LayerNorm forward (float16)");
  m.def("rope_", &rope_cuda, "RoPE in place on q and k (float16)");
}
