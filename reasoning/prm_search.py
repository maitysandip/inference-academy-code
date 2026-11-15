"""Process reward models and tree search on a toy task.

Task: reach a target number from a start number in exactly D steps, each step one of
'+3', '*2', '-1'. A "policy" proposes steps at random (a weak model). We compare:
  - outcome reward (ORM):  sample whole solutions, keep one only if it ends on the target
  - process reward (PRM):  score every partial solution by whether the target is still reachable,
                           and use that score to prune during beam search
A PRM is a learned model in practice; here it is exact (a reachability check) plus noise, so we can see
how its quality changes the result. The budget is counted in policy step proposals.
"""
import random
from functools import lru_cache

OPS = {"+3": lambda x: x + 3, "*2": lambda x: x * 2, "-1": lambda x: x - 1}
START, TARGET, D = 2, 25, 6


@lru_cache(maxsize=None)
def reachable(x, steps):
    if steps == 0:
        return x == TARGET
    return any(reachable(f(x), steps - 1) for f in OPS.values())


def prm(x, steps_left, noise, r):
    """Probability-like score that this partial solution can still succeed. noise=0 is a perfect PRM."""
    true = 1.0 if reachable(x, steps_left) else 0.0
    return true if r.random() >= noise else 1.0 - true


def best_of_n(n, r):
    cost = 0
    for _ in range(n):
        x = START
        for _ in range(D):
            x = OPS[r.choice(list(OPS))](x)
            cost += 1
        if x == TARGET:
            return True, cost
    return False, cost


def beam(width, expand, noise, r):
    beams, cost = [START], 0
    for depth in range(D):
        cands = []
        for x in beams:
            for _ in range(expand):
                y = OPS[r.choice(list(OPS))](x)
                cost += 1
                cands.append((prm(y, D - depth - 1, noise, r), r.random(), y))
        cands.sort(reverse=True)
        beams = [y for _, _, y in cands[:width]]
    return any(x == TARGET for x in beams), cost


def run(fn, trials=2000, seed=0):
    r = random.Random(seed)
    res = [fn(r) for _ in range(trials)]
    return sum(ok for ok, _ in res) / trials, sum(c for _, c in res) / trials


if __name__ == "__main__":
    print(f"task: {START} -> {TARGET} in exactly {D} steps using {list(OPS)}; random policy\n")
    print(f"{'method':<34}{'success':>8}{'avg step proposals':>20}")
    for n in (1, 10, 50, 200):
        acc, cost = run(lambda r: best_of_n(n, r))
        print(f"{'best-of-' + str(n) + ' with outcome check':<34}{acc:>8.3f}{cost:>20.0f}")
    for noise in (0.0, 0.1, 0.3):
        acc, cost = run(lambda r: beam(4, 3, noise, r))
        print(f"{'beam 4x3, PRM noise ' + str(noise):<34}{acc:>8.3f}{cost:>20.0f}")
