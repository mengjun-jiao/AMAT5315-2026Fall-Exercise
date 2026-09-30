"""Automated numerical checks for Week 5 Part 1A."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import part1a_ad


TOLERANCE = 1e-12


def assert_close(actual: float, expected: float) -> None:
    assert abs(actual - expected) <= TOLERANCE, (actual, expected)


def test_part1a_values() -> None:
    results = part1a_ad.as_python(part1a_ad.calculate())
    expected_energy = -0.6570169144600471
    expected_derivative = 2.239979929791143
    expected_a_bar = -2.3425903117359734

    assert_close(results["energy"], expected_energy)
    assert_close(results["tangents"]["U"], expected_derivative)
    assert_close(results["adjoints"]["r"], expected_derivative)
    assert_close(results["jax_grad"], expected_derivative)
    assert_close(results["adjoints"]["a"], expected_a_bar)


def test_forward_reverse_and_jax_grad_agree() -> None:
    results = part1a_ad.as_python(part1a_ad.calculate())
    forward = results["tangents"]["U"]
    reverse = results["adjoints"]["r"]
    reference = results["jax_grad"]
    assert_close(forward, reverse)
    assert_close(forward, reference)
    assert_close(reverse, reference)


if __name__ == "__main__":
    test_part1a_values()
    test_forward_reverse_and_jax_grad_agree()
    print("Part 1A tests passed")
