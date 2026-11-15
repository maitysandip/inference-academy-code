"""Streaming LLM output with Server-Sent Events (SSE), standard library only.

A fake model emits tokens with a fixed prefill delay and a per-token delay. The server streams them as SSE
`data:` lines in the OpenAI chat-completions chunk shape and ends with `data: [DONE]`. The client parses the
stream, measures time to first token (TTFT) against total time, and shows that closing the connection early
stops generation on the server (so you stop paying for tokens nobody reads).
"""
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFILL_S, PER_TOKEN_S = 0.30, 0.03
TOKENS = ("Refunds are accepted within 7 days of purchase, as long as less than 20% of the course "
          "has been opened. Duplicate purchases are refunded at any time.").split(" ")
generated = {"count": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the demo output clean
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")  # tell nginx-style proxies not to buffer the stream
        self.end_headers()
        time.sleep(PREFILL_S)
        try:
            for i, tok in enumerate(TOKENS):
                chunk = {"choices": [{"index": 0, "delta": {"content": tok + (" " if i < len(TOKENS) - 1 else "")}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                generated["count"] += 1
                time.sleep(PER_TOKEN_S)
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away: stop generating


def stream(port, stop_after=None):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=json.dumps({"stream": True}), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    ttft, text, n = None, [], 0
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
        if ttft is None:
            ttft = time.perf_counter() - t0
        text.append(delta); n += 1
        if stop_after and n >= stop_after:
            conn.close()  # user pressed Stop
            break
    return ttft, time.perf_counter() - t0, "".join(text), n


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    ttft, total, text, n = stream(port)
    print(f"full answer : {n} tokens, TTFT {ttft:.2f} s, total {total:.2f} s")
    print(f"              without streaming the user would stare at a blank screen for {total:.2f} s")

    generated["count"] = 0
    ttft, total, text, n = stream(port, stop_after=5)
    time.sleep(0.3)
    print(f"stopped     : client read {n} tokens, server generated {generated['count']} of {len(TOKENS)} before noticing")
    srv.shutdown()
