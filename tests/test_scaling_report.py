import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarking"))

import scaling_report as sr  # noqa: E402

ARGS = dict(params_b=14.7, bytes_per_param=2, layers=48, hidden=5120, batch=1,
            hbm_gbs=3350, hbm_eff=0.8, link_gbs=400, allreduce_us=10)


class TestScalingReport(unittest.TestCase):
    def test_memory_term_divides_by_tp_and_comm_appears(self):
        t1, m1, c1 = sr.decode_step(1, **ARGS)
        t8, m8, c8 = sr.decode_step(8, **ARGS)
        self.assertAlmostEqual(m8, m1 / 8)
        self.assertEqual(c1, 0.0)
        self.assertGreater(c8, 0.0)
        self.assertLess(t8, t1)                 # still faster, but not 8x
        self.assertLess(t1 / t8, 8)

    def test_latency_dominates_small_batches(self):
        # At batch 1 the message is tiny, so per-call latency is most of the communication time.
        _, _, c = sr.decode_step(8, **ARGS)
        latency_only = 2 * ARGS["layers"] * ARGS["allreduce_us"] * 1e-6
        self.assertGreater(latency_only / c, 0.9)

    def test_report_efficiency(self):
        d = tempfile.mkdtemp()
        files = []
        for gpus, tput in ((1, 1000.0), (2, 1800.0)):
            p = os.path.join(d, f"tp{gpus}.json")
            with open(p, "w") as f:
                json.dump({"output_throughput": tput, "median_ttft_ms": 100, "p99_ttft_ms": 300,
                           "median_tpot_ms": 20, "p99_tpot_ms": 40, "request_goodput": 3.0}, f)
            files.append(f"tp{gpus}:{gpus}:{p}")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rows = sr.report(files)
        self.assertEqual(len(rows), 2)
        self.assertIn(" 90%", out.getvalue())   # 1800 / 2 GPUs = 900 per GPU = 90% of 1000


if __name__ == "__main__":
    unittest.main()
