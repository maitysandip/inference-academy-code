// Hopper's Tensor Memory Accelerator (TMA): one thread moves a whole 2D tile.
//
//   nvcc -O3 -std=c++17 -arch=sm_90a tma_tile_copy.cu -o tma_tile_copy -lcuda && ./tma_tile_copy
//
// The host describes the global tensor once, in a CUtensorMap (shape, strides, tile box).
// In the kernel, a single thread asks the TMA unit to copy a tile into shared memory and
// tells an mbarrier how many bytes to expect; every thread waits on the barrier, then the
// block works on the tile (here: multiply by 2) and one thread asks the TMA to write it back.
// No address arithmetic per thread, no per-thread loads: the copy engine does it.
// Follows the TMA example in NVIDIA's CUDA C++ Programming Guide. Needs an H100/H200 (sm_90).
#include <cuda.h>
#include <cuda/barrier>
#include <cstdio>
#include <cstdlib>
#include <vector>

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

constexpr int ROWS = 4096, COLS = 4096;  // global matrix, float32, row major
constexpr int BOX_R = 64, BOX_C = 64;    // one tile: 64 x 64 floats = 16 KiB

__global__ void scale_tiles(const __grid_constant__ CUtensorMap map) {
  __shared__ alignas(128) float tile[BOX_R][BOX_C];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  const int col0 = blockIdx.x * BOX_C, row0 = blockIdx.y * BOX_R;  // TMA coordinates: innermost first
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    cde::fence_proxy_async_shared_cta();  // make the barrier visible to the async (TMA) proxy
  }
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(&tile, &map, col0, row0, bar);
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile));  // arrive, and expect this many bytes
  } else {
    token = bar.arrive();
  }
  bar.wait(std::move(token));  // returns when all threads arrived and all bytes landed

  for (int i = threadIdx.x; i < BOX_R * BOX_C; i += blockDim.x) tile[i / BOX_C][i % BOX_C] *= 2.0f;

  cde::fence_proxy_async_shared_cta();  // make our shared-memory writes visible to the TMA
  __syncthreads();
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(&map, col0, row0, &tile);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();  // shared memory may not be released before the read
  }
  if (threadIdx.x == 0) (&bar)->~barrier();
}

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

int main() {
  std::vector<float> h((size_t)ROWS * COLS);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (float)(i % 1000);
  float* d;
  CHECK(cudaMalloc(&d, h.size() * 4));
  CHECK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));

  // Describe the tensor to the TMA: dimensions and strides are listed innermost first.
  CUtensorMap map;
  cuuint64_t dims[2] = {COLS, ROWS};
  cuuint64_t strides[1] = {COLS * sizeof(float)};  // bytes between rows (the innermost stride is implied)
  cuuint32_t box[2] = {BOX_C, BOX_R};
  cuuint32_t elem_strides[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, d, dims, strides, box,
                                      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("cuTensorMapEncodeTiled failed: %d\n", (int)r); return 1; }

  scale_tiles<<<dim3(COLS / BOX_C, ROWS / BOX_R), 128>>>(map);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  std::vector<float> out(h.size());
  CHECK(cudaMemcpy(out.data(), d, out.size() * 4, cudaMemcpyDeviceToHost));
  size_t bad = 0;
  for (size_t i = 0; i < out.size(); ++i) bad += out[i] != 2.0f * h[i];
  printf("%zu of %zu values wrong\n", bad, out.size());
  return bad != 0;
}
