"""Automated validation for Week 5 Part 1B."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import part1b_compare


def test_grid_and_step() -> None:
    r_values = part1b_compare.separation_values()
    assert r_values.shape == (601,)
    assert float(r_values[0]) == part1b_compare.R_MIN
    assert float(r_values[-1]) == part1b_compare.R_MAX
    assert part1b_compare.FINITE_DIFFERENCE_STEP == 1e-6


def test_derivative_errors_and_zero_crossing() -> None:
    results = part1b_compare.compare()
    summary = part1b_compare.summary(results)
    assert summary["max_forward_absolute_error"] < 1e-12
    assert summary["max_reverse_absolute_error"] < 1e-12
    assert summary["max_finite_difference_absolute_error"] > summary[
        "max_forward_absolute_error"
    ]
    assert summary["max_finite_difference_absolute_error"] > summary[
        "max_reverse_absolute_error"
    ]
    assert 1e-10 < summary["max_finite_difference_absolute_error"] < 1e-7
    assert abs(summary["zero_sample"] - summary["zero_target"]) <= 0.0013


if __name__ == "__main__":
    test_grid_and_step()
    test_derivative_errors_and_zero_crossing()
    print("Part 1B tests passed")
