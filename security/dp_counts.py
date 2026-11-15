"""Differential privacy in one screen: the Laplace mechanism on a count query.

Question: "how many users asked about topic X this week?" Releasing the exact count can reveal whether
one particular person is in the data (compare the count with and without them). The Laplace mechanism
adds noise with scale = sensitivity / epsilon. A count changes by at most 1 when one person is added or
removed, so sensitivity = 1. Smaller epsilon = more privacy = more noise.
"""
import math
import random


def laplace(rng, scale):
    u = rng.random() - 0.5
    return -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))


def noisy_count(true_count, epsilon, rng, sensitivity=1.0):
    return true_count + laplace(rng, sensitivity / epsilon)


if __name__ == "__main__":
    rng = random.Random(7)
    true = 412
    print(f"true count: {true}\n")
    print(f"{'epsilon':>8} {'noise scale':>12} {'5 noisy releases':>40} {'mean abs error (10k runs)':>27}")
    for eps in (0.1, 0.5, 1.0, 2.0):
        samples = [round(noisy_count(true, eps, rng)) for _ in range(5)]
        err = sum(abs(noisy_count(true, eps, rng) - true) for _ in range(10_000)) / 10_000
        print(f"{eps:>8} {1/eps:>12.1f} {str(samples):>40} {err:>27.1f}")

    print("\nComposition: answering the same query k times at epsilon each spends k * epsilon in total.")
    for k in (1, 10, 100):
        print(f"  {k:>3} releases at eps=0.5 -> total budget {k*0.5:g}")

    print("\nWith vs without one person (eps=0.5), 200k releases each:")
    a = [noisy_count(true, 0.5, rng) for _ in range(200_000)]
    b = [noisy_count(true - 1, 0.5, rng) for _ in range(200_000)]
    for t in (410, 415):
        pa, pb = sum(x > t for x in a) / len(a), sum(x > t for x in b) / len(b)
        print(f"  P(release > {t}): with them {pa:.3f}, without them {pb:.3f}, ratio {pa/pb:.2f}")
    print(f"  differential privacy promises the ratio never exceeds e^0.5 = {math.exp(0.5):.2f} (up to sampling noise here)")
