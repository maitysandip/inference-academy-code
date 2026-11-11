import math, os, random, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("llm_from_scratch", "inference_engine", "reasoning", "benchmarking"):
    sys.path.insert(0, os.path.join(ROOT, d))

import bpe, attention, sampling, rope_lora, quant, batching, bench_stats, roofline, methods, engine, gguf_mini


class TestTokenizer(unittest.TestCase):
    def test_round_trip_and_compression(self):
        t = bpe.BPE(); text = "low lower lowest low low newer newest wider widest wide " * 20
        ids = t.train(text, 280)
        self.assertLess(len(ids), len(text.encode()) / 5)
        for s in ["lowest widest", "naïve café 🙂", ""]:
            self.assertEqual(t.decode(t.encode(s)), s)


class TestAttention(unittest.TestCase):
    def test_causality(self):
        random.seed(0); d, T = 4, 5
        rnd = lambda r, c: [[random.gauss(0, .5) for _ in range(c)] for _ in range(r)]
        Wq, Wk, Wv = rnd(d, d), rnd(d, d), rnd(d, d)
        X = [[random.gauss(0, 1) for _ in range(d)] for _ in range(T)]
        A = attention.causal_attention(X, Wq, Wk, Wv)
        X2 = [r[:] for r in X]; X2[4] = [9.0] * d
        B = attention.causal_attention(X2, Wq, Wk, Wv)
        self.assertTrue(all(A[t] == B[t] for t in range(4)))
        self.assertNotEqual(A[4], B[4])

    def test_online_softmax_matches_reference(self):
        random.seed(1); d = 8
        K = [[random.gauss(0, 1) for _ in range(d)] for _ in range(7)]
        V = [[random.gauss(0, 1) for _ in range(d)] for _ in range(7)]
        q = [random.gauss(0, 1) for _ in range(d)]
        ref = attention.attention_ref(q, K, V)
        for block in (1, 2, 3, 7):
            got = attention.attention_online(q, K, V, block)
            self.assertLess(max(abs(a - b) for a, b in zip(ref, got)), 1e-9)


class TestSampling(unittest.TestCase):
    def test_filters(self):
        p = sampling.softmax([4.0, 3.0, 1.0, 0.5, -1.0])
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertEqual([round(x, 3) for x in sampling.top_k_filter(p, 2)], [0.731, 0.269, 0, 0, 0])
        self.assertEqual([round(x, 3) for x in sampling.top_p_filter(p, 0.9)], [0.731, 0.269, 0, 0, 0])


class TestRopeLora(unittest.TestCase):
    def test_rope_relative(self):
        q = [0.3, -1.2, 0.8, 0.5]; k = [1.1, 0.4, -0.6, 0.9]
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))
        self.assertAlmostEqual(dot(rope_lora.rope(q, 5), rope_lora.rope(k, 2)), dot(rope_lora.rope(q, 105), rope_lora.rope(k, 102)), places=9)

    def test_lora_merge(self):
        random.seed(0); d, r = 6, 2
        W = [[random.gauss(0, 1) for _ in range(d)] for _ in range(d)]
        A = [[random.gauss(0, .1) for _ in range(d)] for _ in range(r)]
        B = [[random.gauss(0, .1) for _ in range(r)] for _ in range(d)]
        x = [random.gauss(0, 1) for _ in range(d)]
        y1 = rope_lora.lora_output(W, A, B, x); y2 = rope_lora.matvec(rope_lora.lora_merge(W, A, B), x)
        self.assertLess(max(abs(a - b) for a, b in zip(y1, y2)), 1e-12)


class TestMicroGPT(unittest.TestCase):
    def test_autograd_and_initial_loss(self):
        import microgpt as g
        a, b = g.Value(1.3), g.Value(-0.7); y = (a * b + a * a) * (b * b); y.backward()
        eps = 1e-6; f = lambda x: (x * -0.7 + x * x) * 0.49
        self.assertAlmostEqual(a.grad, (f(1.3 + eps) - f(1.3 - eps)) / (2 * eps), places=4)
        self.assertEqual(len(g.PARAMS), 4128)
        self.assertLess(abs(g.loss_for(g.NAMES[0]).data - math.log(g.VOCAB)), 0.3)


class TestEngine(unittest.TestCase):
    def test_kv_cache_equals_recompute(self):
        cfg = engine.Config(); W = engine.init_weights(cfg); prompt = [3, 14, 15, 9, 2, 6]
        fast = engine.generate(cfg, W, prompt, 6); seq = prompt[:]
        for _ in range(6):
            c = engine.KVCache(cfg)
            for pos, t in enumerate(seq): lg = engine.forward_token(cfg, W, t, pos, c)
            seq.append(engine.argmax(lg))
        self.assertEqual(fast, seq[len(prompt):])

    def test_gguf_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.gguf")
            gguf_mini.write_gguf(p, {"general.architecture": "tiny", "n": 2}, {"a": ((3, 2), [1, 2, 3, 4, 5, 6]), "b": ((2,), [.5, -.5])})
            version, meta, names, tensor = gguf_mini.read_gguf(p)
            self.assertEqual((version, meta, names), (3, {"general.architecture": "tiny", "n": 2}, ["a", "b"]))
            self.assertEqual(tensor("a"), ((3, 2), [1, 2, 3, 4, 5, 6])); self.assertEqual(tensor("b"), ((2,), [.5, -.5]))


class TestQuant(unittest.TestCase):
    def test_int8_error_small_and_int4_larger(self):
        random.seed(1); W = [[random.gauss(0, .05) for _ in range(128)] for _ in range(64)]; x = [random.gauss(0, 1) for _ in range(128)]
        y = quant.matvec_fp(W, x); Q, S = quant.quantize_rows(W); y8 = quant.matvec_q8(Q, S, x); y4 = quant.matvec_fp(quant.q4_group(W), x)
        rms = lambda a, b: math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)) / len(a)); mag = math.sqrt(sum(v * v for v in y) / len(y))
        self.assertLess(rms(y, y8) / mag, 0.02); self.assertGreater(rms(y, y4), rms(y, y8))


class TestBatching(unittest.TestCase):
    def test_continuous_beats_static(self):
        rng = random.Random(3); reqs = [(i * 2, rng.choice([5, 10, 20, 80])) for i in range(60)]
        ls, ts = batching.simulate(reqs, 8, False); lc, tc = batching.simulate(reqs, 8, True)
        self.assertLess(lc, ls); self.assertGreater(tc, ts)


class TestReasoning(unittest.TestCase):
    def test_pass_at_k(self):
        self.assertEqual([round(methods.pass_at_k(20, 5, k), 3) for k in (1, 2, 5, 10)], [0.25, 0.447, 0.806, 0.984])

    def test_voting_helps_or_hurts_depending_on_errors(self):
        diverse = methods.majority_accuracy(0.4, [0.4, 0.3, 0.3], 21, trials=4000)
        trap = methods.majority_accuracy(0.4, [1.0], 21, trials=4000)
        self.assertGreater(diverse, 0.65); self.assertLess(trap, 0.3)

    def test_grpo_toy_learns(self):
        h = methods.grpo_toy(); self.assertAlmostEqual(h[0], 0.2); self.assertGreater(h[-1], 0.95)


class TestBenchmarking(unittest.TestCase):
    def test_open_loop_exposes_queueing(self):
        svc = bench_stats.service_times(4000, 7)
        self.assertGreater(bench_stats.percentile(bench_stats.open_loop(svc, 12), 99), 5 * bench_stats.percentile(svc, 99))

    def test_roofline_h100_decode(self):
        t, m, c = roofline.step_time(8e9, 2, 1, 3.35e12, 989e12)
        self.assertAlmostEqual(1 / t, 209, delta=1)


if __name__ == "__main__":
    unittest.main()
