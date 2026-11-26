// A shared-memory tiled matmul, then the same kernel with a multi-stage cp.async pipeline.
//
//   nvcc -O3 -std=c++17 -arch=sm_80 pipelined_matmul.cu -o pipelined_matmul && ./pipelined_matmul
//
// Both kernels compute C = A x B in float32 with 64x64 output tiles, K tiles of 16 and
// 4x4 outputs per thread (256 threads). The only difference is how tiles reach shared memory:
//   tiled_sync:      every thread loads, waits (__syncthreads), computes, waits again.
//   tiled_pipelined: copies for tile k+STAGES-1 are issued with cp.async while tile k is
//                    computed, so memory latency hides behind arithmetic.
// cp.async needs compute capability 8.0 (Ampere) or newer for real asynchrony; on older GPUs
// the same code compiles but the copies are synchronous, so expect no gain on a T4.
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int BM = 64, BN = 64, BK = 16, THREADS = 256, STAGES = 3;

__device__ inline void compute_tile(const float (*As)[BK], const float (*Bs)[BN], float acc[4][4]) {
  const int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
#pragma unroll
  for (int k = 0; k < BK; ++k) {
    float a[4], b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] = As[ty * 4 + i][k];
#pragma unroll
    for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx * 4 + j];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
  }
}

__device__ inline void store_tile(float* C, int N, float acc[4][4]) {
  const int ty = threadIdx.x / 16, tx = threadIdx.x % 16;
  const int row0 = blockIdx.y * BM + ty * 4, col0 = blockIdx.x * BN + tx * 4;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) C[(size_t)(row0 + i) * N + col0 + j] = acc[i][j];
}

// Requires M % 64 == 0, N % 64 == 0, K % 16 == 0 (kept simple on purpose).
__global__ void tiled_sync(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ __align__(16) float As[BM][BK];
  __shared__ __align__(16) float Bs[BK][BN];
  const int t = threadIdx.x;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < K; k0 += BK) {
    // each thread loads one float4 of A (64x16 tile) and one float4 of B (16x64 tile)
    *reinterpret_cast<float4*>(&As[t / 4][(t % 4) * 4]) =
        *reinterpret_cast<const float4*>(&A[(size_t)(blockIdx.y * BM + t / 4) * K + k0 + (t % 4) * 4]);
    *reinterpret_cast<float4*>(&Bs[t / 16][(t % 16) * 4]) =
        *reinterpret_cast<const float4*>(&B[(size_t)(k0 + t / 16) * N + blockIdx.x * BN + (t % 16) * 4]);
    __syncthreads();                 // tile is in shared memory
    compute_tile(As, Bs, acc);
    __syncthreads();                 // everyone is done before the tile is overwritten
  }
  store_tile(C, N, acc);
}

__global__ void tiled_pipelined(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ __align__(16) float As[STAGES][BM][BK];
  __shared__ __align__(16) float Bs[STAGES][BK][BN];
  const int t = threadIdx.x;
  const int ktiles = K / BK;

  auto issue = [&](int stage, int kt) {  // start (not finish) the copies for K tile kt
    const int k0 = kt * BK;
    __pipeline_memcpy_async(&As[stage][t / 4][(t % 4) * 4],
                            &A[(size_t)(blockIdx.y * BM + t / 4) * K + k0 + (t % 4) * 4], 16);
    __pipeline_memcpy_async(&Bs[stage][t / 16][(t % 16) * 4],
                            &B[(size_t)(k0 + t / 16) * N + blockIdx.x * BN + (t % 16) * 4], 16);
  };

  // Prologue: put STAGES-1 tiles in flight. Commit even when there is nothing to copy, so
  // the number of committed groups stays in step with the loop below.
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ktiles) issue(s, s);
    __pipeline_commit();
  }

  float acc[4][4] = {};
  for (int kt = 0; kt < ktiles; ++kt) {
    __pipeline_wait_prior(STAGES - 2);  // this thread's copies for tile kt have landed
    __syncthreads();                    // ...and everyone else's; also, tile kt-1 is no longer being read
    const int next = kt + STAGES - 1;   // refill the stage that tile kt-1 used
    if (next < ktiles) issue(next % STAGES, next);
    __pipeline_commit();
    compute_tile(As[kt % STAGES], Bs[kt % STAGES], acc);
  }
  store_tile(C, N, acc);
}

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
  const int M = 2048, N = 2048, K = 2048;
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
  srand(0);
  for (auto& v : hA) v = rand() / (float)RAND_MAX - 0.5f;
  for (auto& v : hB) v = rand() / (float)RAND_MAX - 0.5f;

  float *A, *B, *C;
  CHECK(cudaMalloc(&A, hA.size() * 4)); CHECK(cudaMalloc(&B, hB.size() * 4)); CHECK(cudaMalloc(&C, hC.size() * 4));
  CHECK(cudaMemcpy(A, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(B, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice));

  dim3 grid(N / BN, M / BM);
  cudaEvent_t e0, e1;
  CHECK(cudaEventCreate(&e0)); CHECK(cudaEventCreate(&e1));
  const char* names[2] = {"tiled_sync", "tiled_pipelined"};
  for (int which = 0; which < 2; ++which) {
    auto run = [&] {
      if (which == 0) tiled_sync<<<grid, THREADS>>>(A, B, C, M, N, K);
      else tiled_pipelined<<<grid, THREADS>>>(A, B, C, M, N, K);
    };
    run(); CHECK(cudaDeviceSynchronize());            // warm-up
    CHECK(cudaEventRecord(e0));
    for (int r = 0; r < 20; ++r) run();
    CHECK(cudaEventRecord(e1)); CHECK(cudaEventSynchronize(e1));
    float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1)); ms /= 20;

    CHECK(cudaMemcpy(hC.data(), C, hC.size() * 4, cudaMemcpyDeviceToHost));
    double max_err = 0;                                // check 256 random entries on the CPU
    for (int s = 0; s < 256; ++s) {
      int i = rand() % M, j = rand() % N; double ref = 0;
      for (int k = 0; k < K; ++k) ref += (double)hA[(size_t)i * K + k] * hB[(size_t)k * N + j];
      max_err = fmax(max_err, fabs(ref - hC[(size_t)i * N + j]));
    }
    printf("%-16s %8.3f ms  %7.0f GFLOP/s  max error %.2e\n", names[which], ms,
           2.0 * M * N * K / ms / 1e6, max_err);
  }
  return 0;
}
