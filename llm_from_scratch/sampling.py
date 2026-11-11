"""Sampling strategies: temperature, top-k, top-p."""
import math

def softmax(xs, temperature=1.0):
    xs = [x / temperature for x in xs]; m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [v / s for v in e]

def top_k_filter(probs, k):
    keep = set(sorted(range(len(probs)), key=lambda i: -probs[i])[:k])
    p = [pr if i in keep else 0.0 for i, pr in enumerate(probs)]; s = sum(p); return [v / s for v in p]

def top_p_filter(probs, top_p):
    order = sorted(range(len(probs)), key=lambda i: -probs[i]); keep, c = set(), 0.0
    for i in order:
        keep.add(i); c += probs[i]
        if c >= top_p: break
    p = [pr if i in keep else 0.0 for i, pr in enumerate(probs)]; s = sum(p); return [v / s for v in p]
