import json, os, sys, threading, time, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarking"))
import loadgen

DELAY = 0.05  # time to first token built into the fake server


class FakeStream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(DELAY)
        for _ in range(10):
            self.wfile.write(b'data: {"x":1}\n\n'); self.wfile.flush(); time.sleep(0.005)
        self.wfile.write(b"data: [DONE]\n\n")


class TestLoadgen(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(loadgen.percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(loadgen.percentile([0, 10], 95), 9.5)

    def test_measures_ttft_and_goodput_against_a_fake_server(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeStream)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
            res = loadgen.run(url, "m", rate=20, duration=1.0, warmup=0, prompt_lens=[10], out_lens=[10])
            s = loadgen.summarise(res, slo_ttft=0.5, slo_tpot=0.1, duration=1.0, warmup=0)
        finally:
            srv.shutdown()
        self.assertEqual(s["errors"], 0)
        self.assertGreater(s["requests"], 5)
        self.assertGreaterEqual(s["ttft_p50"], DELAY)       # latency includes the built-in delay
        self.assertLess(s["ttft_p50"], DELAY + 0.2)
        self.assertAlmostEqual(s["goodput_req_per_s"], s["requests"], delta=0.01)  # all within SLO over 1 s
        json.dumps(s)


if __name__ == "__main__":
    unittest.main()
