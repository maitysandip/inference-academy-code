"""Lab: an agent loop over MCP tools with guards, a trace, and a trajectory check.

The "model" is scripted (a list of decisions per task) so the lab runs offline and deterministically.
Replace `scripted_model` with a call to your local server's chat endpoint using the same message format.
Guards: a step limit, a repeated-identical-call detector, a token budget and an allow-list of tools.
"""
import json
import sys

from mcp_minimal import Client, PROTOCOL_VERSION

SCRIPTS = {
    "refund": [
        {"tool": "search_notes", "args": {"query": "refund"}},
        {"answer": "Refunds are accepted within 7 days of purchase [search_notes]."},
    ],
    "looping": [{"tool": "search_notes", "args": {"query": "warranty"}}] * 5,
    "forbidden": [{"tool": "delete_everything", "args": {}}, {"answer": "Done."}],
    "maths": [
        {"tool": "add", "args": {"a": 1200, "b": 345}},
        {"answer": "The total is 1545."},
    ],
}
EXPECTED = {  # trajectory expectations used by the check at the end
    "refund": {"tools": ["search_notes"], "must_contain": "7 days"},
    "maths": {"tools": ["add"], "must_contain": "1545"},
}
ALLOWED = {"search_notes", "add"}


def scripted_model(task, step):
    s = SCRIPTS[task]
    return s[step] if step < len(s) else {"answer": "I could not finish."}


def run(client, task, max_steps=4, token_budget=600):
    trace, seen, used = [], set(), 0
    for step in range(max_steps):
        d = scripted_model(task, step)
        used += 60  # rough tokens per model turn (prompt growth + reply) for the budget check
        if used > token_budget:
            return "stopped: token budget", trace
        if "answer" in d:
            trace.append(("answer", d["answer"]))
            return d["answer"], trace
        key = (d["tool"], json.dumps(d["args"], sort_keys=True))
        if d["tool"] not in ALLOWED:
            trace.append(("blocked", d["tool"]))
            return f"stopped: tool {d['tool']} is not allowed", trace
        if key in seen:
            trace.append(("repeat", d["tool"]))
            return "stopped: repeated identical call", trace
        seen.add(key)
        r = client.request("tools/call", {"name": d["tool"], "arguments": d["args"]})
        text = r["result"]["content"][0]["text"] if "result" in r else r["error"]["message"]
        trace.append(("tool", d["tool"], text[:50]))
    return "stopped: step limit", trace


if __name__ == "__main__":
    c = Client([sys.executable, "mcp_minimal.py", "--server"])
    c.request("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "agent-lab", "version": "0.1"}})
    c.notify("notifications/initialized")
    passed = 0
    for task in SCRIPTS:
        result, trace = run(c, task)
        print(f"[{task}] {result}")
        for t in trace:
            print("    ", t)
        if task in EXPECTED:
            e = EXPECTED[task]
            tools = [t[1] for t in trace if t[0] == "tool"]
            ok = tools == e["tools"] and e["must_contain"] in result
            passed += ok
            print(f"    trajectory check: {'PASS' if ok else 'FAIL'} (tools {tools})")
    print(f"\n{passed}/{len(EXPECTED)} trajectory checks passed")
    c.close()
