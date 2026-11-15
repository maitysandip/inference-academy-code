import json, random
random.seed(8)

TOOLS = {
    "get_weather":   {"city": {"type": str}, "unit": {"type": str, "enum": ["c", "f"]}},
    "create_ticket": {"title": {"type": str}, "priority": {"type": str, "enum": ["low", "medium", "high"]}},
    "search_orders": {"order_id": {"type": int}},
}

def validate(call):
    """Return (ok, reason) for one model output string."""
    try: obj = json.loads(call)
    except json.JSONDecodeError: return False, "not JSON"
    if not isinstance(obj, dict) or set(obj) != {"tool", "arguments"}: return False, "wrong shape"
    if obj["tool"] == "none": return (True, "ok") if obj["arguments"] == {} else (False, "none must have empty arguments")
    spec = TOOLS.get(obj["tool"])
    if spec is None: return False, "unknown tool"
    args = obj["arguments"]
    if set(args) != set(spec): return False, "wrong argument names"
    for name, rule in spec.items():
        if not isinstance(args[name], rule["type"]) or isinstance(args[name], bool): return False, f"bad type for {name}"
        if "enum" in rule and args[name] not in rule["enum"]: return False, f"{name} not in enum"
    return True, "ok"

# A test set: requests with the gold answer (a call, or "none" when no tool applies)
CITIES = ["Pune", "Oslo", "Lima", "Cairo", "Seoul", "Quito"]
def make_case():
    k = random.choice(["weather", "ticket", "order", "chat"])
    if k == "weather":
        c, u = random.choice(CITIES), random.choice(["c", "f"]); return f"What's the weather in {c} in {'celsius' if u=='c' else 'fahrenheit'}?", {"tool": "get_weather", "arguments": {"city": c, "unit": u}}
    if k == "ticket":
        t, p = random.choice(["printer is broken", "cannot log in", "invoice is wrong"]), random.choice(["low", "medium", "high"]); return f"Open a {p} priority ticket: {t}", {"tool": "create_ticket", "arguments": {"title": t, "priority": p}}
    if k == "order":
        n = random.randint(1000, 9999); return f"Where is order {n}?", {"tool": "search_orders", "arguments": {"order_id": n}}
    return random.choice(["Thanks, that's all.", "What can you do?", "Tell me a joke."]), {"tool": "none", "arguments": {}}
cases = [make_case() for _ in range(400)]

# Stand-in models: each makes mistakes at the rates below. Replace `ask` with a real model call to evaluate a real model.
PROFILES = {"base model": dict(chatter=.12, wrong_tool=.08, bad_args=.15, spurious=.25),
            "fine-tuned": dict(chatter=.01, wrong_tool=.03, bad_args=.06, spurious=.05),
            "fine-tuned + grammar": dict(chatter=.0, wrong_tool=.03, bad_args=.06, spurious=.05)}
def ask(profile, gold):
    p = PROFILES[profile]; r = random.random(); g = json.loads(json.dumps(gold))
    if r < p["chatter"]: return "Sure! I would call: " + json.dumps(g)
    if g["tool"] == "none":
        if random.random() < p["spurious"]: return json.dumps({"tool": "get_weather", "arguments": {"city": "Pune", "unit": "c"}})
        return json.dumps(g)
    if random.random() < p["wrong_tool"]: return json.dumps({"tool": random.choice([t for t in TOOLS if t != g["tool"]]), "arguments": g["arguments"]})
    if random.random() < p["bad_args"]:
        key = random.choice(list(g["arguments"])); g["arguments"][key] = "high!" if key == "priority" else (str(g["arguments"][key]) if key == "order_id" else g["arguments"][key] + "x")
    return json.dumps(g)

print(f"{'model':<22}{'valid JSON':>11}{'schema ok':>10}{'right tool':>11}{'exact call':>11}{'false calls':>12}")
for name in PROFILES:
    random.seed(1); n = len(cases); js = sch = tool = exact = spurious_calls = no_tool_cases = 0
    for text, gold in cases:
        out = ask(name, gold)
        try: obj = json.loads(out); js += 1
        except json.JSONDecodeError: obj = None
        ok, _ = validate(out); sch += ok
        if obj and obj.get("tool") == gold["tool"]: tool += 1
        exact += obj == gold
        if gold["tool"] == "none":
            no_tool_cases += 1; spurious_calls += bool(obj and obj.get("tool") != "none")
    print(f"{name:<22}{js/n:>11.1%}{sch/n:>10.1%}{tool/n:>11.1%}{exact/n:>11.1%}{spurious_calls/no_tool_cases:>12.1%}")
print("\nvalidator on hand-written examples:")
for ex in ['{"tool": "get_weather", "arguments": {"city": "Oslo", "unit": "c"}}', '{"tool": "get_weather", "arguments": {"city": "Oslo", "unit": "kelvin"}}',
           '{"tool": "search_orders", "arguments": {"order_id": "1234"}}', '{"tool": "delete_everything", "arguments": {}}', 'Sure! {"tool": "none", "arguments": {}}', '{"tool": "none", "arguments": {}}']:
    print(f"  {validate(ex)!s:<42} {ex[:70]}")
