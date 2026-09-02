import random


def estimate_pi(n, seed):
    rng = random.Random(seed)
    points_inside_circle = 0

    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1:
            points_inside_circle += 1

    return 1 * points_inside_circle / n
