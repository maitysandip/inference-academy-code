"""Correctness tests for the Triton kernel pack (code/gpu_kernels/).

Needs PyTorch and Triton. Triton installs on Linux only. Without a GPU, the tests run the
kernels in Triton's interpreter (TRITON_INTERPRET=1, set below), which is slow but exact
enough to catch indexing bugs. They are skipped when torch or triton is missing.
"""
import os
import sys
import unittest

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAVE_CUDA = False
if torch is not None and not HAVE_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")
try:
    import triton  # noqa: F401
except ImportError:
    triton = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpu_kernels"))
DEV = "cuda" if HAVE_CUDA else "cpu"


@unittest.skipIf(torch is None or triton is None, "needs torch and triton (Linux)")
class TestKernelPack(unittest.TestCase):
    def test_rope_matches_reference(self):
        from rope import rope_, rope_reference
        torch.manual_seed(0)
        q = torch.randn(7, 8, 64, device=DEV)
        k = torch.randn(7, 2, 64, device=DEV)                  # grouped-query attention: fewer K heads
        pos = torch.tensor([0, 1, 2, 5, 17, 300, 4095], device=DEV)
        rq, rk = rope_reference(q, k, pos)
        rope_(q, k, pos)
        # At position 4095 the angle is about 4,000 radians, so float32 rounding in how the
        # frequency is computed (exp in the kernel, pow in the reference) shows up near 1e-4.
        torch.testing.assert_close(q, rq, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(k, rk, atol=1e-3, rtol=1e-3)

    def test_rope_position_zero_is_identity(self):
        from rope import rope_
        q = torch.randn(1, 2, 32, device=DEV); k = torch.randn(1, 1, 32, device=DEV)
        q0, k0 = q.clone(), k.clone()
        rope_(q, k, torch.zeros(1, dtype=torch.int64, device=DEV))
        torch.testing.assert_close(q, q0); torch.testing.assert_close(k, k0)

    def test_layernorm_matches_reference(self):
        from layernorm import layernorm, layernorm_reference
        torch.manual_seed(0)
        for n in (64, 1000, 4096):                              # 1000 is not a power of two: exercises the mask
            x = torch.randn(5, n, device=DEV) * 3 + 1
            w = torch.randn(n, device=DEV); b = torch.randn(n, device=DEV)
            torch.testing.assert_close(layernorm(x, w, b), layernorm_reference(x, w, b), atol=1e-4, rtol=1e-4)

    def test_w4_quantise_roundtrip_error_is_bounded(self):
        from w4a16_matmul import quantise_w4, dequantise_w4
        torch.manual_seed(0)
        w = torch.randn(256, 64)
        packed, scales = quantise_w4(w, group=128)
        self.assertEqual(packed.shape, (256, 32)); self.assertEqual(scales.shape, (2, 64))
        err = (dequantise_w4(packed, scales) - w).abs()
        # Rounding error is at most half a step, and a step is the group's scale.
        self.assertTrue(bool((err <= scales.float().repeat_interleave(128, 0) * 0.5 + 1e-3).all()))

    def test_w4a16_matches_dequantised_matmul(self):
        from w4a16_matmul import quantise_w4, w4a16_matmul, w4a16_reference
        torch.manual_seed(0)
        for M in (1, 5, 33):                                    # decode-sized and ragged batches
            x = torch.randn(M, 256, device=DEV).half()
            packed, scales = quantise_w4(torch.randn(256, 192), group=128)
            packed, scales = packed.to(DEV), scales.to(DEV)
            y = w4a16_matmul(x, packed, scales)
            torch.testing.assert_close(y.float(), w4a16_reference(x, packed, scales).float(), atol=2e-1, rtol=2e-2)

    def test_moe_matches_dense_reference(self):
        from moe import moe_linear, moe_reference, align_by_expert, route
        torch.manual_seed(0)
        T, K, N, E, top_k = 37, 64, 96, 8, 2
        x = torch.randn(T, K, device=DEV)
        w = torch.randn(E, K, N, device=DEV) * 0.1
        logits = torch.randn(T, E, device=DEV)
        logits[:, 7] = -1e4                                     # expert 7 gets no tokens: an empty group
        torch.testing.assert_close(moe_linear(x, w, logits, top_k), moe_reference(x, w, logits, top_k),
                                   atol=1e-3, rtol=1e-3)
        _, ids = route(logits, top_k)
        sorted_ids, experts = align_by_expert(ids, E, 16)
        self.assertEqual(sorted_ids.numel() % 16, 0)
        real = sorted_ids[sorted_ids < T * top_k]
        self.assertEqual(sorted(real.tolist()), list(range(T * top_k)))   # every assignment exactly once
        for blk, e in enumerate(experts.tolist()):             # every row in a block belongs to that expert
            rows = sorted_ids[blk * 16:(blk + 1) * 16]
            rows = rows[rows < T * top_k].long()
            self.assertTrue(bool((ids.reshape(-1)[rows] == e).all()))

    def test_sampling_greedy_and_reproducible(self):
        from sampling import sample
        torch.manual_seed(0)
        logits = torch.randn(4, 3000, device=DEV)
        temp = torch.tensor([0.0, 0.0, 1.0, 0.7], device=DEV)
        seeds = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=DEV)
        a, b = sample(logits, temp, seeds), sample(logits, temp, seeds)
        self.assertTrue(bool((a == b).all()))                  # same seeds, same tokens
        self.assertTrue(bool((a[:2] == logits[:2].argmax(-1)).all()))

    def test_sampling_follows_the_softmax(self):
        from sampling import sample
        V, draws, T = 6, 4000, 0.8
        base = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0, 0.3], device=DEV)
        logits = base.repeat(draws, 1)
        seeds = torch.arange(draws, dtype=torch.int32, device=DEV)
        got = torch.bincount(sample(logits, torch.full((draws,), T, device=DEV), seeds, block=8), minlength=V)
        expected = torch.softmax(base / T, dim=0) * draws
        # Each count is binomial; allow five standard deviations.
        sd = (expected * (1 - expected / draws)).sqrt()
        self.assertTrue(bool(((got.float() - expected).abs() <= 5 * sd + 1).all()), (got, expected))


if __name__ == "__main__":
    unittest.main()
