import json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_POST(self):
        n=int(self.headers.get("Content-Length",0)); body=json.loads(self.rfile.read(n))
        self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.end_headers()
        time.sleep(0.05)
        for i in range(min(body.get("max_tokens",20),20)):
            self.wfile.write(b'data: {"x":1}\n\n'); self.wfile.flush(); time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")
ThreadingHTTPServer(("127.0.0.1",8123),H).serve_forever()
