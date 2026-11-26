import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "production"))

import break_even as be  # noqa: E402


class TestBreakEven(unittest.TestCase):
    def test_gpu_count_includes_spares_and_rounds_up(self):
        # 3,000 tok/s per GPU at 60% = 1,800 usable; 2,000 tok/s average needs 2 GPUs + 1 spare.
        tokens = 2000 * be.SECONDS_PER_MONTH
        self.assertEqual(be.gpus_needed(tokens, 3000, 0.6, 1), 3)

    def test_break_even_is_a_crossing(self):
        p = dict(be.DEFAULTS)
        t = be.break_even(p)
        self.assertIsNotNone(t)
        self.assertGreaterEqual(be.api_monthly(t, p), be.self_host_monthly(t, p)[0])
        below = t * 0.95
        self.assertLess(be.api_monthly(below, p), be.self_host_monthly(below, p)[0])

    def test_never_breaks_even_when_marginal_cost_is_above_api(self):
        p = dict(be.DEFAULTS, gpu_tok_s=2000, utilisation=0.5)   # $0.80 per million vs $0.60
        self.assertGreater(be.marginal_cost_per_million(p), p["api_price"])
        self.assertIsNone(be.break_even(p))

    def test_cheaper_gpus_break_even_sooner(self):
        p = dict(be.DEFAULTS)
        cheaper = dict(p, gpu_price=p["gpu_price"] * 0.7)
        self.assertLess(be.break_even(cheaper), be.break_even(p))


if __name__ == "__main__":
    unittest.main()
