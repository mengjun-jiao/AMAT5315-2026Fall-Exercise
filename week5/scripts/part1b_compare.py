"""Compare analytic, hand-written AD, and finite-difference derivatives."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from part1a_ad import energy, forward_pass, reverse_pass


R_MIN = 0.95
R_MAX = 2.5
SAMPLE_COUNT = 601
FINITE_DIFFERENCE_STEP = 1e-6


def separation_values() -> Any:
    """Return the required 601-point inclusive separation grid."""
    return jnp.linspace(R_MIN, R_MAX, SAMPLE_COUNT, dtype=jnp.float64)


def analytic_derivative(r_values: Any) -> Any:
    """Evaluate the analytic Lennard-Jones derivative."""
    return 24 * (r_values**-7 - 2 * r_values**-13)


def forward_derivative(r_values: Any) -> Any:
    """Reuse the Part 1A node-by-node forward-mode implementation."""
    return jax.vmap(lambda r: forward_pass(r)["U"])(r_values)


def reverse_derivative(r_values: Any) -> Any:
    """Reuse the Part 1A node-by-node reverse-mode implementation."""
    return jax.vmap(lambda r: reverse_pass(r)["r"])(r_values)


def finite_difference_derivative(
    r_values: Any, step: float = FINITE_DIFFERENCE_STEP
) -> Any:
    """Evaluate the centered finite-difference derivative of the Part 1A energy."""
    return (energy(r_values + step) - energy(r_values - step)) / (2 * step)


def compare() -> dict[str, Any]:
    """Compute all derivative methods and their absolute errors."""
    r_values = separation_values()
    analytic = analytic_derivative(r_values)
    forward = forward_derivative(r_values)
    reverse = reverse_derivative(r_values)
    finite_difference = finite_difference_derivative(r_values)
    errors = {
        "forward": jnp.abs(forward - analytic),
        "reverse": jnp.abs(reverse - analytic),
        "finite_difference": jnp.abs(finite_difference - analytic),
    }
    zero_target = 2 ** (1 / 6)
    zero_index = int(jnp.argmin(jnp.abs(r_values - zero_target)))
    return {
        "r": r_values,
        "analytic": analytic,
        "forward": forward,
        "reverse": reverse,
        "finite_difference": finite_difference,
        "errors": errors,
        "zero_target": zero_target,
        "zero_index": zero_index,
    }


def summary(results: dict[str, Any]) -> dict[str, float]:
    """Extract the required numerical validation summary."""
    r_values = results["r"]
    zero_index = results["zero_index"]
    return {
        "max_forward_absolute_error": float(jnp.max(results["errors"]["forward"])),
        "max_reverse_absolute_error": float(jnp.max(results["errors"]["reverse"])),
        "max_finite_difference_absolute_error": float(
            jnp.max(results["errors"]["finite_difference"])
        ),
        "zero_target": float(results["zero_target"]),
        "zero_sample": float(r_values[zero_index]),
        "zero_sample_derivative": float(results["analytic"][zero_index]),
    }


def plot_comparison(results: dict[str, Any], output_path: Path) -> None:
    """Create the two-panel derivative and absolute-error comparison figure."""
    r_values = results["r"]
    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, constrained_layout=True)

    axes[0].plot(r_values, results["analytic"], label="Analytic", linewidth=2)
    axes[0].plot(r_values, results["forward"], label="Forward AD", linestyle="--")
    axes[0].plot(r_values, results["reverse"], label="Reverse AD", linestyle=":")
    axes[0].plot(
        r_values,
        results["finite_difference"],
        label="Centered finite difference",
        linestyle="-.",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_ylabel("dU/dr (reduced units)")
    axes[0].set_title("Lennard-Jones derivative comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    for name, label, style in (
        ("forward", "Forward AD", "--"),
        ("reverse", "Reverse AD", ":"),
        ("finite_difference", "Centered finite difference", "-"),
    ):
        axes[1].plot(
            r_values,
            jnp.maximum(results["errors"][name], 1e-16),
            label=label,
            linestyle=style,
        )
    axes[1].set_xlabel("Separation r (reduced units)")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_yscale("log")
    axes[1].set_title("Absolute error relative to analytic derivative")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, format="png")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ad/modes.png"),
        help="PNG output path, relative to the current directory",
    )
    args = parser.parse_args()
    results = compare()
    plot_comparison(results, args.output)
    for name, value in summary(results).items():
        print(f"{name}: {value:.17g}")


if __name__ == "__main__":
    main()
