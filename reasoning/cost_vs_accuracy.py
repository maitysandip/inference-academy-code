"""Cost-versus-accuracy simulation: compare reasoning strategies on problems of varying difficulty."""
import random
from collections import Counter

def make_problems(n, seed=0):
    rng = random.Random(seed)
    # Each problem has a per-sample success probability (its 'easiness'), drawn from a Beta distribution: many easy, some hard.
    return [rng.betavariate(1.2, 1.0) for _ in range(n)]

def sample_answer(p, rng, n_wrong=4):
    """One sampled chain: correct w.p. p, otherwise one of n_wrong wrong answers (the first is a tempting trap, 40% of wrong mass)."""
    if rng.random() < p: return "right"
    return rng.choices([f"w{i}" for i in range(n_wrong)], weights=[0.4, 0.2, 0.2, 0.2])[0]

def direct(problems, rng, tokens=20, skill=0.55):
    wins = sum(rng.random() < p * skill for p in problems); return wins / len(problems), tokens
def cot(problems, rng, tokens=400):
    wins = sum(sample_answer(p, rng) == "right" for p in problems); return wins / len(problems), tokens
def vote(problems, rng, n=9, tokens=400):
    wins = 0
    for p in problems:
        c = Counter(sample_answer(p, rng) for _ in range(n)); top = max(c.values())
        wins += rng.choice([a for a, v in c.items() if v == top]) == "right"
    return wins / len(problems), tokens * n
def best_of_n(problems, rng, n=9, tokens=400, verifier_acc=0.9, verifier_cost=30):
    wins = 0
    for p in problems:
        cands = []
        for _ in range(n):
            a = sample_answer(p, rng); ok = (a == "right")
            cands.append((a, rng.random() < (verifier_acc if ok else 1 - verifier_acc)))
        good = [a for a, g in cands if g] or [a for a, _ in cands]
        wins += rng.choice(good) == "right"
    return wins / len(problems), (tokens + verifier_cost) * n
def adaptive_vote(problems, rng, first=3, max_n=15, tokens=400):
    """Sample 3; if they all agree stop, else keep sampling up to max_n (spend compute only on hard problems)."""
    wins, spent = 0, 0
    for p in problems:
        answers = [sample_answer(p, rng) for _ in range(first)]; spent += first * tokens
        if len(set(answers)) > 1:
            extra = [sample_answer(p, rng) for _ in range(max_n - first)]; answers += extra; spent += (max_n - first) * tokens
        c = Counter(answers); top = max(c.values())
        wins += rng.choice([a for a, v in c.items() if v == top]) == "right"
    return wins / len(problems), spent / len(problems)

if __name__ == "__main__":
    rng = random.Random(42); P = make_problems(3000)
    rows = [("direct answer", direct(P, rng)), ("chain of thought", cot(P, rng)), ("CoT + vote (n=9)", vote(P, rng)),
            ("CoT + vote (n=21)", vote(P, rng, n=21)), ("best-of-9, verifier 0.9", best_of_n(P, rng)),
            ("adaptive vote (3→15)", adaptive_vote(P, rng))]
    print(f"{'method':26s} {'accuracy':>9s} {'tokens/problem':>15s}")
    for name, (acc, tok) in rows: print(f"{name:26s} {acc:9.3f} {tok:15.0f}")
