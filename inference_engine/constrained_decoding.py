"""Grammar-constrained decoding: mask the logits so only tokens that keep the output valid can be sampled.

The grammar here is a tiny JSON object: {"sentiment": "pos"|"neg"|"neutral", "score": <1-2 digit int>}.
It is compiled by hand into a character-level automaton. At each step the engine:
  1. computes which vocabulary tokens keep the partial output a valid prefix of the grammar,
  2. sets every other token's logit to -inf,
  3. samples from what is left.
The "model" is a random scorer that prefers chatty tokens, so unconstrained it rarely produces valid JSON.
Real engines (llama.cpp GBNF, Outlines, XGrammar, llguidance) do the same with precompiled token masks.
"""
import json
import math
import random
import re

VOCAB = ['{', '}', '"', ':', ',', ' ', 'sentiment', 'score', 'pos', 'neg', 'neutral', 'itive', '0', '1', '2', '7', '9',
         '42', 'Sure', '!', ' Here', ' is', ' the', ' JSON', '\n', '```', 'json', '<eos>']
FULL = re.compile(r'\{"sentiment": ?"(pos|neg|neutral)", ?"score": ?\d{1,2}\}')
PREFIX_TARGETS = [f'{{"sentiment": "{s}", "score": {n}}}' for s in ("pos", "neg", "neutral") for n in range(100)]


def is_valid_prefix(text):
    # exact for this tiny grammar: the text must be a prefix of some valid string (we use canonical spacing)
    return any(t.startswith(text) for t in PREFIX_TARGETS)


def allowed(text):
    ok = [t for t in VOCAB if t != '<eos>' and is_valid_prefix(text + t)]
    if FULL.fullmatch(text):
        ok.append('<eos>')
    return ok


def fake_logits(r):
    # a chatty model: prefers prose and code fences, slightly
    bias = {'Sure': 2.0, ' Here': 1.5, '```': 1.5, '\n': 1.0, '!': 1.0}
    return {t: r.gauss(0, 1) + bias.get(t, 0) for t in VOCAB}


def sample(logits, r, temp=1.0):
    m = max(logits.values())
    w = {t: math.exp((v - m) / temp) for t, v in logits.items() if v != -math.inf}
    x = r.random() * sum(w.values())
    for t, v in w.items():
        x -= v
        if x <= 0:
            return t
    return t


def generate(r, constrained, max_tokens=24):
    text = ""
    for _ in range(max_tokens):
        logits = fake_logits(r)
        if constrained:
            ok = set(allowed(text))
            logits = {t: (v if t in ok else -math.inf) for t, v in logits.items()}
        t = sample(logits, r)
        if t == '<eos>':
            break
        text += t
    return text


def valid(s):
    try:
        return FULL.fullmatch(s) is not None and isinstance(json.loads(s), dict)
    except Exception:
        return False


if __name__ == "__main__":
    r = random.Random(3)
    print("unconstrained samples:")
    for _ in range(3):
        print("  ", repr(generate(r, False)))
    print("constrained samples:")
    for _ in range(3):
        print("  ", generate(r, True))
    n = 500
    u = sum(valid(generate(r, False)) for _ in range(n))
    c = sum(valid(generate(r, True)) for _ in range(n))
    print(f"\nvalid JSON matching the schema: unconstrained {u}/{n}, constrained {c}/{n}")
    print("allowed next tokens after '{\"sentiment\": \"':", allowed('{"sentiment": "'))
    print("allowed next tokens after '{\"sentiment\": \"neg\", \"score\": 4':", allowed('{"sentiment": "neg", "score": 4'))
