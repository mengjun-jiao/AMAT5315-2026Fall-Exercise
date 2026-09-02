The function estimate_pi(n, seed) estimates pi by generating n random points
uniformly in the unit square from (0, 0) to (1, 1), counting the fraction whose
distance from the origin is at most 1, and returning four times that fraction.
The seed fixes the random-number sequence so that the same inputs produce the
same result. The definition of correct is:

abs(estimate_pi(1_000_000, seed=2026) - math.pi) < 1e-2

