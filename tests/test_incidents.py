import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "incidents"))

import incident_sim as sim  # noqa: E402
import scenarios  # noqa: E402


class CheckedReplica(sim.Replica):
    """Checks after every step that no KV block is lost or counted twice."""
    def step(self):
        ran = super().step()
        held = sum(r.own_blocks for r in self.running)
        cached = sum(b for b, _, _ in self.pool.cache.values())
        assert held + cached + self.pool.free == self.pool.total, (held, cached, self.pool.free, self.pool.total)
        assert all(r.computed <= r.prefill_target() + 1 for r in self.running)
        return ran


class TestEngineModel(unittest.TestCase):
    def test_memory_plan(self):
        p = sim.Engine().memory_plan()
        self.assertAlmostEqual(p["budget_gb"], 43.2)
        self.assertAlmostEqual(p["kv_gb"], 5.5, places=6)
        self.assertEqual(p["kv_tokens"], 20976)
        self.assertEqual(sim.Engine(kv_cache_dtype="fp8").memory_plan()["kv_tokens"], 41952)
        self.assertEqual(sim.Engine(enforce_eager=True).memory_plan()["graphs_gb"], 0.0)

    def test_step_time_roofline(self):
        e = sim.Engine()
        decode = e.step_seconds(kv_tokens_read=0, tokens_computed=1)
        self.assertAlmostEqual(decode, 0.003 + 35 / (864 * 0.8), places=6)   # memory-bound: read the weights once
        self.assertGreater(e.step_seconds(0, 2048), 0.9)                       # compute-bound prefill chunk

    def test_startup_fails_when_cache_smaller_than_max_model_len(self):
        with self.assertRaises(SystemExit):
            sim.Replica(sim.Engine(gpu_memory_utilization=0.80), sim.Workload())

    def test_block_accounting_under_pressure(self):
        for load in (replace(scenarios.LOADTEST, clients=40, duration_s=300), replace(scenarios.CHAT, header_position="start", duration_s=300)):
            r = CheckedReplica(sim.Engine(), load, seed=3).run()
            self.assertGreater(r["completed"], 0)


class TestScenarios(unittest.TestCase):
    def test_ttft_scenario_and_fix(self):
        sc = scenarios.ttft()
        a = sim.simulate(sc.before_engine, sc.before_load)
        b = sim.simulate(sc.now_engine, sc.now_load)
        self.assertLess(a["ttft_p50"], 0.5)
        self.assertGreater(b["ttft_p50"], 5 * a["ttft_p50"])
        self.assertLess(b["prefix_hit_rate"], 0.05)
        self.assertGreater(a["prefix_hit_rate"], 0.5)
        self.assertAlmostEqual(b["prompt_mean"] / a["prompt_mean"], 1.0, delta=0.03)
        fixed = sim.simulate(sc.now_engine, replace(sc.now_load, header_position="end"))
        self.assertLess(fixed["ttft_p50"], 0.5)

    def test_memory_scenario_fp8_fix(self):
        sc = scenarios.memory()
        peak = sim.simulate(sc.now_engine, sc.now_load)
        fp8 = sim.simulate(replace(sc.now_engine, kv_cache_dtype="fp8"), sc.now_load)
        self.assertGreater(peak["ttft_p99"], 20)
        self.assertLess(fp8["ttft_p99"], peak["ttft_p99"] / 4)
        self.assertGreater(peak["kv_usage"], 0.7)

    def test_concurrency_collapse_and_admission_control(self):
        sc = scenarios.concurrency()
        peak = sim.simulate(sc.before_engine, sc.before_load)["out_tok_s"]
        high = sim.simulate(sc.now_engine, sc.now_load)
        self.assertLess(high["out_tok_s"], 0.7 * peak)
        self.assertGreater(high["gen_tok_s_window"], 0.9 * peak)      # the GPU is still busy: the work is wasted
        gated = sim.simulate(sc.now_engine, replace(sc.now_load, gateway_limit=12))
        self.assertGreater(gated["out_tok_s"], 0.9 * peak)

    def test_blind_seeds_cover_every_fault(self):
        n = len(scenarios.BLIND_FAULTS)
        for start in (5, 10):
            names = {scenarios.blind(s).answer.split("]")[0] for s in range(start, start + n)}
            self.assertEqual(len(names), n)


if __name__ == "__main__":
    unittest.main()
