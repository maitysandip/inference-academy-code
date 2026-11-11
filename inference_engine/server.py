import json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from engine import Config, init_weights, KVCache, forward_token, argmax
cfg, W = Config(), init_weights(Config())
LOCK = threading.Lock()          # one request at a time: a real engine batches instead (next lesson)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        if self.path != "/v1/completions": self.send_error(404); return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt, n = body["prompt"], int(body.get("max_tokens", 8))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        with LOCK:
            cache, logits = KVCache(cfg), None
            for pos, t in enumerate(prompt): logits = forward_token(cfg, W, t, pos, cache)
            for i in range(n):
                tok = argmax(logits)
                chunk = {"choices": [{"index": 0, "token": tok}]}
                try: self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode()); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError): return          # client went away: stop generating
                logits = forward_token(cfg, W, tok, len(prompt) + i, cache)
            self.wfile.write(b"data: [DONE]\n\n")
if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8124), H).serve_forever()
