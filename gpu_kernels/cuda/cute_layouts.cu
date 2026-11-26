// CuTe layouts on the host: the algebra CUTLASS 3 and 4 kernels are built from.
//
//   nvcc -std=c++17 -I<cutlass>/include cute_layouts.cu -o cute_layouts && ./cute_layouts
//
// A CuTe Layout is a pair (shape, stride): a function from logical coordinates to an offset.
// Everything else (tensors, tiles, how threads and tensor-core instructions are mapped to
// data) is built by composing and dividing layouts. This file runs on the CPU; no GPU needed.
#include <cute/tensor.hpp>
#include <cstdio>
using namespace cute;

int main() {
  // 1. A 4x8 matrix, column major (CuTe's default) and row major.
  auto col_major = make_layout(make_shape(Int<4>{}, Int<8>{}));
  auto row_major = make_layout(make_shape(Int<4>{}, Int<8>{}), LayoutRight{});
  printf("column major: "); print(col_major); printf("\n");
  printf("row major:    "); print(row_major); printf("\n");
  printf("element (1,2) is at offset %d (column major) and %d (row major)\n\n",
         int(col_major(1, 2)), int(row_major(1, 2)));

  // 2. Every offset of the row-major layout, as a picture.
  print_layout(row_major);

  // 3. Tiling: split an 8x8 row-major matrix into 4x4 tiles. The result is a layout whose
  //    first mode walks inside a tile and whose second mode walks between tiles.
  auto m8 = make_layout(make_shape(Int<8>{}, Int<8>{}), LayoutRight{});
  auto tiled = zipped_divide(m8, make_shape(Int<4>{}, Int<4>{}));
  printf("\n8x8 divided into 4x4 tiles: "); print(tiled); printf("\n");
  printf("tile (1,0), element (0,0) is at offset %d\n\n", int(tiled(make_coord(0, 0), make_coord(1, 0))));

  // 4. A tensor is a pointer plus a layout. Fill one through a row-major view, read it back
  //    through a tile view of the same memory.
  float data[64];
  auto t = make_tensor(&data[0], m8);
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) t(i, j) = 10 * i + j;
  auto tile_11 = zipped_divide(t, make_shape(Int<4>{}, Int<4>{}))(make_coord(_, _), make_coord(1, 1));
  printf("tile (1,1) of the 8x8 tensor:\n");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) printf("%4.0f", float(tile_11(i, j)));
    printf("\n");
  }

  // 5. A swizzle: XOR some offset bits so that a column of a shared-memory tile lands in
  //    different banks. Swizzle<3,0,3> XORs the row number (offset bits 3-5) into the column
  //    (bits 0-2), so element (i, 0) moves to column i of row i.
  auto sw = composition(Swizzle<3, 0, 3>{}, make_layout(make_shape(Int<8>{}, Int<8>{}), LayoutRight{}));
  printf("\nrow-major 8x8 with Swizzle<3,0,3>: offsets of column 0, rows 0..7:");
  for (int i = 0; i < 8; ++i) printf(" %d", int(sw(i, 0)));
  printf("\nfloat32 banks (offset %% 32), swizzled:");
  for (int i = 0; i < 8; ++i) printf(" %d", int(sw(i, 0)) % 32);
  printf("\nunswizzled the offsets are 0 8 16 ... 56, which fall in only 4 of the 32 banks\n");
  return 0;
}
