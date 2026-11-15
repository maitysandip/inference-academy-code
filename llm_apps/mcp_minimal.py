"""A minimal Model Context Protocol (MCP) server and client over stdio, standard library only.

MCP messages are JSON-RPC 2.0, one JSON object per line on stdin/stdout for the stdio transport.
This implements the core of the lifecycle: initialize -> notifications/initialized -> tools/list -> tools/call.
It is a teaching subset: real servers should use an official SDK, which also handles resources, prompts,
cancellation, progress, pagination and newer protocol versions.

Run the demo (the client starts the server as a subprocess):
    python3 mcp_minimal.py
Run only the server (for a real MCP host to launch):
    python3 mcp_minimal.py --server
"""
import json
import subprocess
import sys

PROTOCOL_VERSION = "2025-06-18"  # check the current spec; hosts and servers negotiate this in initialize

NOTES = {"refunds": "Refunds are accepted within 7 days of purchase.", "shipping": "Orders ship within 2 working days."}

TOOLS = [
    {
        "name": "search_notes",
        "description": "Search the team's policy notes by keyword and return matching notes.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
    },
]


def call_tool(name, args):
    """Run a tool. Validation errors are returned as tool results with isError, not as protocol errors."""
    if name == "search_notes":
        q = str(args.get("query", "")).lower()
        hits = [f"{k}: {v}" for k, v in sorted(NOTES.items()) if q in k or q in v.lower()]
        return {"content": [{"type": "text", "text": "\n".join(hits) or "No matching notes."}], "isError": False}
    if name == "add":
        a, b = args.get("a"), args.get("b")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return {"content": [{"type": "text", "text": "Both a and b must be numbers."}], "isError": True}
        return {"content": [{"type": "text", "text": str(a + b)}], "isError": False}
    raise KeyError(name)


def handle(msg):
    """Return a response dict for a request, or None for a notification."""
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:  # notifications (e.g. notifications/initialized) get no reply
        return None
    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "notes-server", "version": "0.1.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params", {})
        try:
            result = call_tool(p.get("name"), p.get("arguments", {}))
        except KeyError:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Unknown tool: {p.get('name')}"}}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve():
    # stdout carries protocol messages only; anything else (logs) must go to stderr.
    for line in sys.stdin:
        if not line.strip():
            continue
        reply = handle(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


class Client:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.next_id = 1

    def request(self, method, params=None):
        mid, self.next_id = self.next_id, self.next_id + 1
        self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})
        reply = json.loads(self.p.stdout.readline())
        assert reply["id"] == mid
        return reply

    def notify(self, method):
        self._send({"jsonrpc": "2.0", "method": method})

    def _send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=5)


def demo():
    c = Client([sys.executable, __file__, "--server"])
    init = c.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "demo-host", "version": "0.1.0"}})
    print("server:", init["result"]["serverInfo"]["name"], "protocol", init["result"]["protocolVersion"])
    c.notify("notifications/initialized")
    tools = c.request("tools/list")["result"]["tools"]
    print("tools:", [t["name"] for t in tools])
    r = c.request("tools/call", {"name": "search_notes", "arguments": {"query": "refund"}})
    print("search_notes ->", r["result"]["content"][0]["text"])
    r = c.request("tools/call", {"name": "add", "arguments": {"a": 2, "b": "three"}})
    print("add (bad args) -> isError", r["result"]["isError"], "|", r["result"]["content"][0]["text"])
    r = c.request("tools/call", {"name": "delete_everything", "arguments": {}})
    print("unknown tool -> error", r["error"]["code"], r["error"]["message"])
    c.close()


if __name__ == "__main__":
    serve() if "--server" in sys.argv else demo()
