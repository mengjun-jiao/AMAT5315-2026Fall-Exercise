"""Single-point hand-written forward and reverse AD for Lennard-Jones energy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)


R_VALUE = 1.3


def primal(r: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Evaluate the four specified computational nodes."""
    a = r**-6
    b = a**2
    c = b - a
    energy = 4 * c
    return r, a, b, c, energy


def forward_pass(r: Any, r_dot: Any = 1.0) -> dict[str, Any]:
    """Propagate a tangent through each computational node with local JVPs."""
    r, a, b, c, energy = primal(r)
    _, a_dot = jax.jvp(lambda x: x**-6, (r,), (r_dot,))
    _, b_dot = jax.jvp(lambda x: x**2, (a,), (a_dot,))
    _, c_dot = jax.jvp(lambda x, y: x - y, (b, a), (b_dot, a_dot))
    _, energy_dot = jax.jvp(lambda x: 4 * x, (c,), (c_dot,))
    return {"r": r_dot, "a": a_dot, "b": b_dot, "c": c_dot, "U": energy_dot}


def reverse_pass(r: Any) -> dict[str, Any]:
    """Propagate adjoints backward, explicitly accumulating the shared a node."""
    r, a, b, c, _ = primal(r)
    energy_bar = 1.0

    _, energy_pullback = jax.vjp(lambda x: 4 * x, c)
    (c_bar,) = energy_pullback(energy_bar)

    _, c_pullback = jax.vjp(lambda x, y: x - y, b, a)
    b_bar, a_bar_from_c = c_pullback(c_bar)

    _, b_pullback = jax.vjp(lambda x: x**2, a)
    (a_bar_from_b,) = b_pullback(b_bar)
    a_bar = a_bar_from_c + a_bar_from_b

    _, a_pullback = jax.vjp(lambda x: x**-6, r)
    (r_bar,) = a_pullback(a_bar)

    return {
        "r": r_bar,
        "a": a_bar,
        "b": b_bar,
        "c": c_bar,
        "U": energy_bar,
    }


def energy(r: Any) -> Any:
    """Independent scalar Lennard-Jones energy used only for the JAX check."""
    return 4 * (r**-12 - r**-6)


def calculate(r_value: float = R_VALUE) -> dict[str, Any]:
    """Run both hand-written passes and the independent JAX gradient check."""
    r = jax.numpy.asarray(r_value, dtype=jax.numpy.float64)
    _, _, _, _, energy_value = primal(r)
    return {
        "r": r,
        "energy": energy_value,
        "tangents": forward_pass(r),
        "adjoints": reverse_pass(r),
        "jax_grad": jax.grad(energy)(r),
    }


def as_python(value: Any) -> Any:
    """Convert scalar JAX values and nested mappings into JSON-compatible values."""
    if isinstance(value, dict):
        return {key: as_python(item) for key, item in value.items()}
    return float(value)


def write_evidence(output_path: Path) -> dict[str, Any]:
    results = as_python(calculate())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ad/derivatives.json"),
        help="JSON evidence output path, relative to the current directory",
    )
    args = parser.parse_args()
    results = write_evidence(args.output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
