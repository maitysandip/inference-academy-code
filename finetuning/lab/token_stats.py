import json, random, math
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
data = json.load(open("tickets.json"))["train"]
SYSTEM = 'Classify the support ticket. Reply with JSON only, like {"category": "billing", "priority": "low"}.'
rows = []
for r in data:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["ticket"]}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": json.dumps(r["label"])}], tokenize=True, return_dict=True)["input_ids"]
    rows.append((len(p), len(full) - len(p), len(full)))
P = sum(a for a, _, _ in rows); C = sum(b for _, b, _ in rows); T = sum(c for _, _, c in rows)
print(f"examples {len(rows)}  tokens/example: prompt {P/len(rows):.0f}  completion {C/len(rows):.0f}  total {T/len(rows):.0f}")
print(f"share of tokens that are the answer: {C/T:.1%}   prompt tokens: {P/T:.1%}")
print("example completion tokens:", tok.convert_ids_to_tokens(tok.encode(json.dumps(data[0]['label']))) )

# padding waste for batches of 8
lens = [c for _, _, c in rows]
random.seed(0)
def waste_random(bs=8):
    idx = list(range(len(lens))); random.shuffle(idx)
    pad = real = 0
    for i in range(0, len(idx), bs):
        b = [lens[j] for j in idx[i:i+bs]]; pad += max(b) * len(b); real += sum(b)
    return 1 - real/pad
def waste_sorted(bs=8):
    s = sorted(lens); pad = real = 0
    for i in range(0, len(s), bs):
        b = s[i:i+bs]; pad += max(b) * len(b); real += sum(b)
    return 1 - real/pad
def pack(limit=512):
    bins, cur = 0, 0
    for l in lens:
        if cur + l > limit: bins += 1; cur = 0
        cur += l
    bins += 1
    return 1 - sum(lens)/(bins*limit), bins
print(f"min/mean/max length: {min(lens)}/{sum(lens)/len(lens):.0f}/{max(lens)}")
print(f"padding waste, random batches of 8 : {waste_random():.1%}")
print(f"padding waste, length-grouped      : {waste_sorted():.1%}")
w, b = pack(512); print(f"padding waste, packed to 512 tokens: {w:.1%}  ({b} sequences instead of {len(lens)})")

# bigger variance example: chat data with long tail
random.seed(1)
long = [int(random.lognormvariate(5.2, 0.8)) + 20 for _ in range(2000)]
def waste(ls, bs=16, srt=False):
    ls = sorted(ls) if srt else ls[:]; pad = real = 0
    for i in range(0, len(ls), bs):
        b = ls[i:i+bs]; pad += max(b)*len(b); real += sum(b)
    return 1 - real/pad
print(f"\nheavy-tailed lengths (median {sorted(long)[1000]}, max {max(long)}): random {waste(long):.1%}, grouped {waste(long, srt=True):.1%}")
