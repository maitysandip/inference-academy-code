import math, random
from collections import Counter

# ---- pass@k (unbiased estimator, Chen et al. 2021) ----
def pass_at_k(n, c, k):
    """n samples, c correct: probability that at least one of k drawn without replacement is correct."""
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

# ---- self-consistency: majority vote over n samples ----
def majority_accuracy(p_correct, wrong_dist, n, trials=20000, seed=0):
    rng = random.Random(seed); wins = 0
    answers = ["right"] + [f"w{i}" for i in range(len(wrong_dist))]
    weights = [p_correct] + [(1 - p_correct) * w for w in wrong_dist]
    for _ in range(trials):
        votes = Counter(rng.choices(answers, weights=weights, k=n))
        top = max(votes.values()); leaders = [a for a, v in votes.items() if v == top]
        wins += rng.choice(leaders) == "right"
    return wins / trials

# ---- best-of-n with a noisy verifier ----
def best_of_n(p_correct, verifier_acc, n, trials=20000, seed=1):
    """Each sample is right with p_correct. The verifier scores right answers high w.p. verifier_acc, wrong ones high w.p. 1-verifier_acc."""
    rng = random.Random(seed); wins = 0
    for _ in range(trials):
        cands = []
        for _ in range(n):
            right = rng.random() < p_correct
            looks_good = rng.random() < (verifier_acc if right else 1 - verifier_acc)
            cands.append((looks_good, right))
        good = [r for g, r in cands if g]
        pick = rng.choice(good) if good else rng.choice([r for _, r in cands])
        wins += pick
    return wins / trials

# ---- toy GRPO: a softmax policy over 5 candidate answers, reward 1 if the answer is correct ----
def grpo_toy(steps=60, group=8, lr=0.5, seed=0, correct=3, clip=0.2):
    rng = random.Random(seed); theta = [0.0] * 5
    def probs(th):
        m = max(th); e = [math.exp(t - m) for t in th]; s = sum(e); return [x / s for x in e]
    hist = []
    for step in range(steps):
        p = probs(theta)
        samples = rng.choices(range(5), weights=p, k=group)
        rewards = [1.0 if a == correct else 0.0 for a in samples]
        mean = sum(rewards) / group; std = (sum((r - mean) ** 2 for r in rewards) / group) ** 0.5
        adv = [(r - mean) / (std + 1e-6) for r in rewards]               # group-relative advantage: no value network
        grad = [0.0] * 5
        for a, A in zip(samples, adv):
            for j in range(5): grad[j] += A * ((1.0 if j == a else 0.0) - p[j]) / group   # d log pi(a) / d theta_j
        theta = [t + lr * g for t, g in zip(theta, grad)]
        hist.append(p[correct])
    return hist

if __name__ == "__main__":
    print("pass@k n=20 c=5:", [round(pass_at_k(20, 5, k), 3) for k in (1, 2, 5, 10)])
    for n in (1, 3, 5, 9, 21):
        print("majority n", n, round(majority_accuracy(0.4, [0.4, 0.3, 0.3], n), 3), "| uniform wrong p=.4:", round(majority_accuracy(0.4, [1.0], n), 3))
    for n in (1, 4, 16):
        print("best-of", n, "weak verifier(.7):", round(best_of_n(0.4, 0.7, n), 3), "strong(.95):", round(best_of_n(0.4, 0.95, n), 3))
    h = grpo_toy(); print("GRPO toy p(correct) at steps 0,10,30,59:", [round(h[i], 3) for i in (0, 10, 30, 59)])
