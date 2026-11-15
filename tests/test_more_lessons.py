import os
import random
import sys
import unittest

HERE = os.path.dirname(__file__)
for d in ("security", "evals", "reasoning", "inference_engine", "llm_from_scratch", "finetuning", "multimodal", "production"):
    sys.path.insert(0, os.path.join(HERE, "..", d))

import chat_template  # noqa: E402
import constrained_decoding  # noqa: E402
import cost_latency_eval  # noqa: E402
import dedup  # noqa: E402
import dp_counts  # noqa: E402
import ocr_metrics  # noqa: E402
import precision  # noqa: E402
import prm_search  # noqa: E402
import secure_rag  # noqa: E402
import serverless_vs_dedicated as sv  # noqa: E402
import trajectory_eval  # noqa: E402
import zero_sharding  # noqa: E402


class TestSecurity(unittest.TestCase):
    def test_acl_filters_before_ranking(self):
        ids = [d["id"] for d in secure_rag.retrieve("api key salary refunds", {"support"})]
        self.assertNotIn("eng-1", ids)
        self.assertNotIn("hr-1", ids)

    def test_screen_and_output_check(self):
        kept, quarantined = secure_rag.screen(secure_rag.DOCS)
        self.assertEqual([q[0] for q in quarantined], ["pub-2"])
        self.assertTrue(secure_rag.check_output("key sk-test-51Habc123"))
        self.assertFalse(secure_rag.check_output("see https://academy.example/billing"))

    def test_dp_noise_scale(self):
        rng = random.Random(0)
        err = sum(abs(dp_counts.noisy_count(100, 1.0, rng) - 100) for _ in range(20000)) / 20000
        self.assertAlmostEqual(err, 1.0, delta=0.05)


class TestEvals(unittest.TestCase):
    def test_trajectory_grading(self):
        passed = {k for k, t in trajectory_eval.RUNS.items() if trajectory_eval.score(t, trajectory_eval.TASK)["pass"]}
        self.assertEqual(passed, {"A clean", "D looping"})

    def test_percentile_and_wilson(self):
        self.assertEqual(cost_latency_eval.pct(list(range(1, 101)), 95), 95)
        lo, hi = cost_latency_eval.wilson(80, 100)
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)


class TestReasoningAndEngine(unittest.TestCase):
    def test_prm_reachability(self):
        self.assertTrue(prm_search.reachable(prm_search.START, prm_search.D))

    def test_constrained_always_valid(self):
        r = random.Random(1)
        self.assertTrue(all(constrained_decoding.valid(constrained_decoding.generate(r, True)) for _ in range(50)))
        self.assertEqual(constrained_decoding.allowed('{"sentiment": "'), ["pos", "neg", "neutral"])

    def test_chat_template_generation_prompt(self):
        self.assertTrue(chat_template.chatml(chat_template.MESSAGES).endswith("<|im_start|>assistant\n"))
        ids = chat_template.tokenize(chat_template.llama3(chat_template.MESSAGES), add_bos=False)
        self.assertEqual(ids.count("<|begin_of_text|>"), 1)


class TestBuildAndOps(unittest.TestCase):
    def test_minhash_estimates_jaccard(self):
        a = dedup.shingles(dedup.DOCS["a"])
        c = dedup.shingles(dedup.DOCS["c"])
        sa, sc = dedup.minhash(a, n=256), dedup.minhash(c, n=256)
        est = sum(x == y for x, y in zip(sa, sc)) / 256
        self.assertAlmostEqual(est, dedup.jaccard(a, c), delta=0.1)

    def test_precision(self):
        self.assertEqual(precision.fp16(70000.0), float("inf"))
        self.assertEqual(precision.bf16(1.0 + 1e-4), 1.0)

    def test_zero_sharding(self):
        P = 8e9
        self.assertEqual(zero_sharding.per_gpu(P, 8, 0), 16 * P)
        self.assertAlmostEqual(zero_sharding.per_gpu(P, 8, 3), 2 * P)

    def test_ocr_metrics(self):
        self.assertEqual(ocr_metrics.edit_distance("kitten", "sitting"), 3)

    def test_serverless_cost_caps_at_hour(self):
        cost, up = sv.serverless_cost_per_hour(5000, 1)
        self.assertEqual(up, 3600)


if __name__ == "__main__":
    unittest.main()
