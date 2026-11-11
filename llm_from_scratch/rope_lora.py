"""Rotary position embeddings and the LoRA merge identity."""
import math

def rope(x, pos, base=10000.0):
    out = x[:]
    for i in range(0, len(x), 2):
        theta = pos * base ** (-i / len(x))
        c, s = math.cos(theta), math.sin(theta)
        out[i], out[i + 1] = x[i] * c - x[i + 1] * s, x[i] * s + x[i + 1] * c
    return out

def matvec(M, v): return [sum(m * u for m, u in zip(row, v)) for row in M]

def lora_output(W, A, B, x): return [a + b for a, b in zip(matvec(W, x), matvec(B, matvec(A, x)))]

def lora_merge(W, A, B):
    d, r = len(W), len(A)
    return [[W[i][j] + sum(B[i][k] * A[k][j] for k in range(r)) for j in range(len(W[0]))] for i in range(d)]
