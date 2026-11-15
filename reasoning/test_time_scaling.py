"""Test-time scaling: how accuracy grows with samples per problem, and how to spend a fixed budget.

Each problem has its own difficulty: the chance p that one sample is right. Easy problems have p near 1,
hard ones near 0. With majority voting over n samples, accuracy on a problem depends on p (and on how the
wrong answers spread). We compare:
  - uniform:  every problem gets the same n samples
  - adaptive: estimate difficulty from a few probe samples, stop early on easy problems (all probes agree),
              and spend the saved budget on the uncertain ones
at the same AVERAGE number of samples per problem.
"""
import random
from collections import Counter

rng = random.Random(0)
N_PROBLEMS = 2000
# difficulty mix: 50% easy, 30% medium, 20% hard
PROBLEMS = [rng.choice([0.92, 0.95, 0.97]) for _ in range(1000)] + [rng.choice([0.5, 0.6, 0.7]) for _ in range(600)] + [rng.choice([0.15, 0.25, 0.35]) for _ in range(400)]
WRONG = 4  # distinct wrong answers, equally likely


def sample(p, r):
    return "right" if r.random() < p else f"w{r.randrange(WRONG)}"


def vote(answers, r):
    c = Counter(answers)
    top = max(c.values())
    return r.choice(sorted(a for a, v in c.items() if v == top))


def uniform(n, seed=1):
    r = random.Random(seed)
    right = sum(vote([sample(p, r) for _ in range(n)], r) == "right" for p in PROBLEMS)
    return right / len(PROBLEMS), n


def adaptive(probe, extra, seed=1):
    r = random.Random(seed)
    right, used = 0, 0
    for p in PROBLEMS:
        ans = [sample(p, r) for _ in range(probe)]
        if len(set(ans)) > 1:  # disagreement: spend more on this problem
            ans += [sample(p, r) for _ in range(extra)]
        used += len(ans)
        right += vote(ans, r) == "right"
    return right / len(PROBLEMS), used / len(PROBLEMS)


if __name__ == "__main__":
    print("uniform majority voting")
    print(f"{'samples/problem':>16} {'accuracy':>9}")
    for n in (1, 3, 5, 9, 17, 33):
        acc, _ = uniform(n)
        print(f"{n:>16} {acc:>9.3f}")

    print("\nadaptive: 3 probe samples, more only when they disagree")
    print(f"{'extra if unsure':>16} {'avg samples':>12} {'accuracy':>9} {'uniform at same budget':>23}")
    for extra in (4, 8, 16):
        acc, avg = adaptive(3, extra)
        uacc, _ = uniform(round(avg))
        print(f"{extra:>16} {avg:>12.1f} {acc:>9.3f} {uacc:>16.3f} (n={round(avg)})")

    print("\nby difficulty at uniform n=9:")
    r = random.Random(5)
    for label, lo, hi in (("easy", 0.9, 1.0), ("medium", 0.4, 0.8), ("hard", 0.0, 0.4)):
        ps = [p for p in PROBLEMS if lo <= p < hi]
        a1 = sum(sample(p, r) == "right" for p in ps) / len(ps)
        a9 = sum(vote([sample(p, r) for _ in range(9)], r) == "right" for p in ps) / len(ps)
        print(f"  {label:<7} n={len(ps):>4}  1 sample {a1:.3f}  -> 9 samples {a9:.3f}")
