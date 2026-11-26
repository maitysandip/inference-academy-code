// A persistent, warp-specialised kernel: one producer warp copies, four consumer warps compute.
//
//   nvcc -O3 -std=c++17 -arch=sm_80 warp_specialized_rowsum.cu -o ws_rowsum && ./ws_rowsum
//
// Task: y[r] = sum of row r of a [ROWS x COLS] float32 matrix (memory bound, like many
// inference kernels). Two versions:
//   rowsum_simple:  one block per row; every thread loads and adds, then a block reduction.
//   rowsum_ws:      a persistent grid (one block per SM) loops over rows. In each block,
//                   warp 0 is the producer: it copies the next rows into a ring of STAGES
//                   shared-memory buffers with cp.async. Warps 1-4 are consumers: they wait
//                   for a full buffer, reduce it and hand the buffer back. The two roles meet
//                   only at the pipeline's acquire/commit/wait/release points, so loading
//                   row i+3 overlaps with reducing row i.
// Consumers synchronise among themselves with a named barrier (bar.sync 1, 128), because a
// __syncthreads() would also wait for the producer warp, which is busy elsewhere.
// This is the structure (much simplified) of Hopper GEMM kernels such as CUTLASS's
// warp-specialised schedules, where the producer uses TMA and the consumers use tensor cores.
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

constexpr int ROWS = 16384, COLS = 4096;  // 256 MiB of float32
constexpr int STAGES = 4;
constexpr int CONSUMERS = 128;            // four consumer warps
constexpr int THREADS = 32 + CONSUMERS;   // plus one producer warp

__global__ void rowsum_simple(const float* __restrict__ x, float* __restrict__ y) {
  __shared__ float red[32];
  const float* row = x + (size_t)blockIdx.x * COLS;
  float s = 0.f;
  for (int i = threadIdx.x; i < COLS; i += blockDim.x) s += row[i];
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x < 32) {
    s = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (threadIdx.x == 0) y[blockIdx.x] = s;
  }
}

__device__ inline void consumer_barrier() { asm volatile("bar.sync 1, %0;" ::"n"(CONSUMERS)); }

__global__ void rowsum_ws(const float* __restrict__ x, float* __restrict__ y) {
  extern __shared__ __align__(16) float buf[];  // STAGES x COLS floats
  __shared__ float red[CONSUMERS / 32];
#pragma nv_diag_suppress static_var_with_dynamic_init  // make_pipeline initialises it
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> state;

  auto block = cg::this_thread_block();
  const bool producer = threadIdx.x < 32;
  auto pipe = cuda::make_pipeline(block, &state, producer ? cuda::pipeline_role::producer
                                                          : cuda::pipeline_role::consumer);

  // Persistent loop: this block handles rows blockIdx.x, blockIdx.x + gridDim.x, ...
  // Producer and consumers walk the same sequence, so iteration i always uses stage i % STAGES.
  int i = 0;
  for (int r = blockIdx.x; r < ROWS; r += gridDim.x, ++i) {
    float* stage = buf + (size_t)(i % STAGES) * COLS;
    if (producer) {
      pipe.producer_acquire();                       // wait until this stage is free
      constexpr int PER_LANE = COLS / 32;            // each lane copies a contiguous slice
      const int lane = threadIdx.x;
      cuda::memcpy_async(stage + lane * PER_LANE, x + (size_t)r * COLS + lane * PER_LANE,
                         cuda::aligned_size_t<16>(PER_LANE * sizeof(float)), pipe);
      pipe.producer_commit();
    } else {
      pipe.consumer_wait();                          // wait until the stage is full
      const int c = threadIdx.x - 32;
      float s = 0.f;
      for (int k = c; k < COLS; k += CONSUMERS) s += stage[k];
      pipe.consumer_release();                       // the producer may refill it now
      for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
      if ((c & 31) == 0) red[c >> 5] = s;
      consumer_barrier();                            // consumers only
      if (c == 0) y[r] = red[0] + red[1] + red[2] + red[3];
      consumer_barrier();                            // red is reused next iteration
    }
  }
}

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
  std::vector<float> hx((size_t)ROWS * COLS), hy(ROWS), ref(ROWS);
  srand(0);
  for (auto& v : hx) v = rand() / (float)RAND_MAX;
  for (int r = 0; r < ROWS; ++r) {
    double s = 0; for (int k = 0; k < COLS; ++k) s += hx[(size_t)r * COLS + k]; ref[r] = (float)s;
  }
  float *x, *y;
  CHECK(cudaMalloc(&x, hx.size() * 4)); CHECK(cudaMalloc(&y, ROWS * 4));
  CHECK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));

  int sms; CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const size_t smem = (size_t)STAGES * COLS * sizeof(float);  // 64 KiB: needs the opt-in below
  CHECK(cudaFuncSetAttribute(rowsum_ws, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  cudaEvent_t e0, e1; CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
  const char* names[2] = {"rowsum_simple", "rowsum_ws"};
  for (int which = 0; which < 2; ++which) {
    auto run = [&] {
      if (which == 0) rowsum_simple<<<ROWS, 256>>>(x, y);
      else rowsum_ws<<<sms, THREADS, smem>>>(x, y);
    };
    run(); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(e0));
    for (int k = 0; k < 20; ++k) run();
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1)); ms /= 20;
    CHECK(cudaMemcpy(hy.data(), y, ROWS * 4, cudaMemcpyDeviceToHost));
    double max_rel = 0;
    for (int r = 0; r < ROWS; ++r) max_rel = fmax(max_rel, fabs(hy[r] - ref[r]) / fabs(ref[r]));
    printf("%-14s %7.3f ms  %6.0f GB/s  max relative error %.1e  (grid %d)\n", names[which], ms,
           hx.size() * 4.0 / ms / 1e6, max_rel, which ? sms : ROWS);
  }
  return 0;
}
