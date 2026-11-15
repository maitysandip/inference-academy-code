import json, random, re, collections
random.seed(4)
LABELS = ["billing", "login", "shipping", "bug", "feature"]
SEEDS = {
 "billing": ["I was charged twice this month", "my invoice total looks wrong", "please refund my last payment", "the coupon did not apply"],
 "login": ["I cannot sign in", "the reset email never arrived", "my two-factor code fails", "my account is locked"],
 "shipping": ["my parcel has not arrived", "tracking has not updated", "the box arrived damaged", "I received the wrong item"],
 "bug": ["the app crashes on startup", "export does nothing", "the page is very slow", "search shows no results"],
 "feature": ["please add dark mode", "can you support CSV import", "I would like weekly reports", "add a team sharing option"],
}
FILLERS = ["", " Thanks.", " Please help.", " It started yesterday.", " This is the second time.", " Using the mobile app."]

# 1. A mock teacher model: it mostly follows the format, but makes the mistakes real generators make.
EXTRA = ["", " on my phone", " on my laptop", " since Monday", " after the update", " for the third time", " in the EU store", " on the team plan"]
history = []
def teacher(i):
    label = random.choices(LABELS, weights=[6, 4, 4, 3, 1])[0]          # skewed: feature requests are rare
    text = random.choice(SEEDS[label]) + random.choice(EXTRA) + random.choice(FILLERS)
    r = random.random()
    if r < 0.04:  return "Sure! Here is a ticket: " + text                                     # not JSON
    if r < 0.07:  return json.dumps({"ticket": text})                                           # missing label
    if r < 0.10:  return json.dumps({"ticket": text, "label": "refunds"})                       # label not in the set
    if r < 0.12:  return json.dumps({"ticket": text[:8], "label": label})                       # too short
    if r < 0.15:  return json.dumps({"ticket": text + f" Contact me at user{i}@example.com or 555-01{i%100:02d}.", "label": label})  # PII
    if r < 0.18:  return json.dumps({"ticket": "As an AI language model, I cannot write tickets. " + text, "label": label})          # refusal
    if r < 0.30 and history:  return random.choice(history)                                     # repeats an earlier output verbatim
    if r < 0.38 and history:                                                                    # near duplicate: same ticket, trivial edits
        d = json.loads(random.choice(history)); d["ticket"] = d["ticket"].lower().rstrip(".") + "!"; return json.dumps(d)
    out = json.dumps({"ticket": text, "label": label}); history.append(out); return out
raw = [teacher(i) for i in range(1200)]

# 2. A held-out evaluation set that must never leak into training
EVAL = ["I was charged twice this month. It started yesterday.", "the reset email never arrived. Thanks.", "my parcel has not arrived. This is the second time."]

stages = [("raw generations", len(raw))]
def keep(name, rows): stages.append((name, len(rows))); return rows

# parse + schema
parsed = []
for line in raw:
    try: parsed.append(json.loads(line))
    except json.JSONDecodeError: pass
rows = keep("valid JSON", parsed)
rows = keep("has required fields", [r for r in rows if isinstance(r, dict) and {"ticket", "label"} <= r.keys()])
rows = keep("label in allowed set", [r for r in rows if r["label"] in LABELS])
# quality heuristics
BAD = re.compile(r"as an ai|language model|i cannot", re.I)
rows = keep("length and refusal filters", [r for r in rows if 15 <= len(r["ticket"]) <= 300 and not BAD.search(r["ticket"])])
# PII: drop anything that looks like an email or phone number
PII = re.compile(r"[\w.+-]+@[\w-]+\.\w+|\b\d{3}-\d{4}\b")
rows = keep("no PII", [r for r in rows if not PII.search(r["ticket"])])
# exact duplicates (after normalising case and spaces)
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip(" .!")
seen, uniq = set(), []
for r in rows:
    k = (norm(r["ticket"]), r["label"])
    if k not in seen: seen.add(k); uniq.append(r)
rows = keep("exact duplicates removed", uniq)
# near duplicates: word-5-gram Jaccard, keep first of each cluster
def shingles(s, n=3): w = norm(s).split(); return {" ".join(w[i:i+n]) for i in range(max(1, len(w)-n+1))}
kept, sh = [], []
for r in rows:
    s = shingles(r["ticket"])
    if all(len(s & t)/len(s | t) < 0.6 for t in sh): kept.append(r); sh.append(s)
rows = keep("near duplicates removed", kept)
# decontamination against the eval set
ev = [shingles(e) for e in EVAL]
rows = keep("eval overlap removed", [r for r in rows if all(len(shingles(r["ticket"]) & e)/len(e) < 0.5 for e in ev)])
before = collections.Counter(r["label"] for r in rows)
# balance: cap the biggest classes at 2x the smallest
cap = 2 * min(before.values()); per = collections.Counter(); bal = []
random.shuffle(rows)
for r in rows:
    if per[r["label"]] < cap: per[r["label"]] += 1; bal.append(r)
rows = keep(f"class balance (cap {cap} per class)", bal)

print(f"{'stage':<36}{'rows':>6}{'removed':>9}")
prev = None
for name, n in stages:
    print(f"{name:<36}{n:>6}{'' if prev is None else prev - n:>9}"); prev = n
print(f"\nyield: {stages[-1][1]/stages[0][1]:.1%} of raw generations survive")
print("classes before balancing:", dict(sorted(before.items())))
print("classes after balancing :", dict(sorted(collections.Counter(r['label'] for r in rows).items())))
print("sample row:", rows[0])
