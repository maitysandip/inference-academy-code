// Matrix-vector product benchmark: float32 vs int8 weights (weights are streamed from memory once per call).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

void matvec_f32(const float *W, const float *x, float *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float acc = 0.f; const float *w = W + (size_t)r * cols;
        for (int c = 0; c < cols; c++) acc += w[c] * x[c];
        y[r] = acc;
    }
}
void matvec_q8(const int8_t *Q, const float *S, const float *x, float *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float acc = 0.f; const int8_t *w = Q + (size_t)r * cols;
        for (int c = 0; c < cols; c++) acc += (float)w[c] * x[c];
        y[r] = acc * S[r];
    }
}
int main(void) {
    int rows = 8192, cols = 8192, reps = 20;                    // 256 MB as float32, 64 MB as int8
    float *W = malloc((size_t)rows * cols * sizeof(float)); int8_t *Q = malloc((size_t)rows * cols);
    float *x = malloc(cols * sizeof(float)), *y = malloc(rows * sizeof(float)), *S = malloc(rows * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * cols; i++) { W[i] = (float)((i * 2654435761u) % 1000) / 1000.f - 0.5f; Q[i] = (int8_t)((i * 40503u) % 255 - 127); }
    for (int i = 0; i < cols; i++) x[i] = 1.f; for (int i = 0; i < rows; i++) S[i] = 0.01f;
    matvec_f32(W, x, y, rows, cols); matvec_q8(Q, S, x, y, rows, cols);   // warm up
    double t0 = now(); for (int i = 0; i < reps; i++) matvec_f32(W, x, y, rows, cols); double tf = (now() - t0) / reps;
    t0 = now(); for (int i = 0; i < reps; i++) matvec_q8(Q, S, x, y, rows, cols); double tq = (now() - t0) / reps;
    printf("float32: %.1f ms  (%.1f GB/s)\n", tf * 1e3, (double)rows * cols * 4 / tf / 1e9);
    printf("int8   : %.1f ms  (%.1f GB/s)  speedup %.2fx\n", tq * 1e3, (double)rows * cols / tq / 1e9, tf / tq);
    printf("checksum %.3f\n", y[0] + y[rows - 1]);
    return 0;
}
