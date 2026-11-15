import json, math, random, re, collections
random.seed(0)
data = json.load(open("predictions.json"))
test, reg = data["test"], data["regression"]

# --- parsing: strict (the whole reply must be JSON) and lenient (find the first {...} in the reply) ---
def strict(text):
    try: return json.loads(text)
    except Exception: return None
def lenient(text):
    m = re.search(r"\{.*?\}", text, re.S)
    return strict(m.group(0)) if m else None
def score(row, who, parse):
    got = parse(row[who]); gold = row["label"]
    if not isinstance(got, dict): return dict(valid=0, category=0, priority=0, exact=0)
    c, p = got.get("category") == gold["category"], got.get("priority") == gold["priority"]
    return dict(valid=1, category=int(c), priority=int(p), exact=int(c and p))

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n
    centre = (p + z*z/(2*n)) / d; half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return centre - half, centre + half

n = len(test)
print(f"{n} held-out tickets")
print(f"{'model / parser':<22}{'valid JSON':>11}{'category':>10}{'priority':>10}{'exact':>8}   95% interval for exact")
S = {}
for who in ("base", "tuned"):
    for pname, parse in (("strict", strict), ("lenient", lenient)):
        rows = [score(r, who, parse) for r in test]; S[(who, pname)] = rows
        tot = {k: sum(r[k] for r in rows) for k in rows[0]}; lo, hi = wilson(tot["exact"], n)
        print(f"{who + ' / ' + pname:<22}{tot['valid']/n:>11.0%}{tot['category']/n:>10.0%}{tot['priority']/n:>10.0%}{tot['exact']/n:>8.0%}   {lo:.0%} to {hi:.0%}")

# --- is the improvement real? paired comparison on exact match, base (lenient) against tuned ---
b = [r["exact"] for r in S[("base", "lenient")]]; t = [r["exact"] for r in S[("tuned", "strict")]]
only_t = sum(1 for x, y in zip(b, t) if y and not x); only_b = sum(1 for x, y in zip(b, t) if x and not y)
m = only_t + only_b
p_value = min(1.0, 2 * sum(math.comb(m, i) for i in range(0, min(only_t, only_b) + 1)) / 2**m) if m else 1.0
diffs = []
for _ in range(5000):
    idx = [random.randrange(n) for _ in range(n)]
    diffs.append(sum(t[i] - b[i] for i in idx) / n)
diffs.sort()
print(f"\npaired comparison, tuned (strict) vs base (lenient): tuned right only on {only_t}, base right only on {only_b}")
print(f"exact McNemar p-value {p_value:.2g}; improvement {sum(t)/n - sum(b)/n:+.0%}, bootstrap 95% interval {diffs[125]:+.0%} to {diffs[4874]:+.0%}")

# --- error analysis for the tuned model ---
errs = collections.Counter(); conf = collections.Counter()
for r in test:
    got = strict(r["tuned"])
    if got and got.get("priority") != r["label"]["priority"]: conf[(r["label"]["priority"], got.get("priority"))] += 1
    if got and got.get("category") != r["label"]["category"]: errs[r["label"]["category"]] += 1
print("\ntuned model, priority confusions (gold -> predicted):", dict(conf.most_common()))
print("tuned model, category mistakes by gold class:", dict(errs) or "none")
for r in [r for r in test if (strict(r['tuned']) or {}).get('priority') != r['label']['priority']][:3]:
    print("  gold", r["label"]["priority"], "| predicted", (strict(r["tuned"]) or {}).get("priority"), "|", r["ticket"][:70])

# --- regression suite: did it get worse at things we did not train? ---
def passed(row, who): return any(k in row[who].lower() for k in row["keywords"])
def jsony(row, who): return row[who].lstrip().startswith(("{", '"'))
print(f"\nregression suite ({len(reg)} general prompts)")
for who in ("base", "tuned"):
    print(f"  {who:<6} answers containing an expected word: {sum(passed(r, who) for r in reg)}/{len(reg)}   replies wrapped in quotes or braces: {sum(jsony(r, who) for r in reg)}/{len(reg)}")
for r in reg:
    if passed(r, "base") and not passed(r, "tuned"): print(f"  regression: {r['prompt'][:46]:<48} base: {r['base'][:34]!r}  tuned: {r['tuned'][:34]!r}")
