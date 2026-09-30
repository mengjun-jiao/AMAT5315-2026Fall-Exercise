"""Benchmark forward- and reverse-mode AD scaling for Lennard-Jones clusters."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


ATOM_COUNTS = (64, 128, 256, 512, 1024)
LATTICE_SPACING = 2 ** (1 / 6)
PERTURBATION_STD = 0.05
RANDOM_SEED = 5315
DEFAULT_REPETITIONS = 5


def scalar_input_count(atom_count: int) -> int:
    """Return the number of Cartesian scalar inputs for N atoms."""
    return 3 * atom_count


def pair_indices(atom_count: int) -> tuple[Any, Any]:
    """Return each unordered pair i<j exactly once."""
    return jnp.triu_indices(atom_count, k=1)


def make_cluster(atom_count: int, seed: int = RANDOM_SEED) -> Any:
    """Construct a deterministic perturbed simple-cubic cluster without PBCs."""
    side = math.ceil(atom_count ** (1 / 3))
    axis = jnp.arange(side, dtype=jnp.float64)
    lattice = jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    lattice = lattice.reshape((-1, 3))[:atom_count] * LATTICE_SPACING
    perturbation_key = jax.random.PRNGKey(seed)
    perturbation = PERTURBATION_STD * jax.random.normal(
        perturbation_key, shape=(atom_count, 3), dtype=jnp.float64
    )
    return lattice + perturbation


def make_cluster_energy(atom_count: int) -> Callable[[Any], Any]:
    """Create a JAX-compatible all-pairs cluster energy function."""
    pair_i, pair_j = pair_indices(atom_count)

    def cluster_energy(coordinates: Any) -> Any:
        displacement = coordinates[pair_i] - coordinates[pair_j]
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=1))
        pair_energy = 4 * (distance**-12 - distance**-6)
        return jnp.sum(pair_energy)

    return cluster_energy


def analytic_gradient(coordinates: Any) -> Any:
    """Compute the independent coordinate gradient by pair accumulation."""
    atom_count = coordinates.shape[0]
    pair_i, pair_j = pair_indices(atom_count)
    displacement = coordinates[pair_i] - coordinates[pair_j]
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=1))
    radial_derivative = 24 * (distance**-7 - 2 * distance**-13)
    pair_gradient = radial_derivative[:, None] * displacement / distance[:, None]
    gradient = jnp.zeros_like(coordinates)
    gradient = gradient.at[pair_i].add(pair_gradient)
    gradient = gradient.at[pair_j].add(-pair_gradient)
    return gradient


def make_flat_energy(cluster_energy: Callable[[Any], Any], atom_count: int) -> Callable[[Any], Any]:
    def flat_energy(flat_coordinates: Any) -> Any:
        return cluster_energy(flat_coordinates.reshape((atom_count, 3)))

    return flat_energy


def make_forward_component(flat_energy: Callable[[Any], Any]) -> Callable[[Any, Any], Any]:
    """JIT one scalar JVP component; callers invoke it once per input direction."""
    def forward_component(flat_coordinates: Any, direction: Any) -> Any:
        return jax.jvp(
            flat_energy,
            (flat_coordinates,),
            (direction,),
        )[1]

    return jax.jit(forward_component)


def make_reverse_gradient(flat_energy: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return jax.jit(jax.grad(flat_energy))


def forward_gradient(
    flat_coordinates: Any,
    forward_component: Callable[[Any, Any], Any],
) -> Any:
    """Assemble a full gradient from P separate one-coordinate JVPs."""
    components = []
    for coordinate_index in range(flat_coordinates.size):
        direction = jnp.zeros_like(flat_coordinates).at[coordinate_index].set(1.0)
        components.append(forward_component(flat_coordinates, direction))
    return jnp.stack(components)


def relative_error(computed: Any, reference: Any) -> float:
    """Use max absolute component error divided by max reference magnitude."""
    numerator = jnp.max(jnp.abs(computed - reference))
    denominator = jnp.max(jnp.abs(reference))
    return float(numerator / denominator)


def synchronize(value: Any) -> None:
    value.block_until_ready()


def median_time(function: Callable[[], Any], repetitions: int) -> float:
    """Measure synchronized execution time and return the median sample."""
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        synchronize(result)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def benchmark_one(atom_count: int, repetitions: int) -> dict[str, float | int]:
    coordinates = make_cluster(atom_count)
    flat_coordinates = coordinates.reshape((-1,))
    cluster_energy = make_cluster_energy(atom_count)
    flat_energy = make_flat_energy(cluster_energy, atom_count)
    energy_jit = jax.jit(flat_energy)
    forward_component = make_forward_component(flat_energy)
    reverse_gradient = make_reverse_gradient(flat_energy)

    reference = analytic_gradient(coordinates)
    synchronize(reference)
    synchronize(energy_jit(flat_coordinates))
    synchronize(reverse_gradient(flat_coordinates))
    synchronize(forward_gradient(flat_coordinates, forward_component))

    energy_time = median_time(lambda: energy_jit(flat_coordinates), repetitions)
    forward_time = median_time(
        lambda: forward_gradient(flat_coordinates, forward_component), repetitions
    )
    reverse_time = median_time(lambda: reverse_gradient(flat_coordinates), repetitions)

    forward_result = forward_gradient(flat_coordinates, forward_component)
    reverse_result = reverse_gradient(flat_coordinates)
    synchronize(forward_result)
    synchronize(reverse_result)
    reference_flat = reference.reshape((-1,))
    return {
        "N": atom_count,
        "P": scalar_input_count(atom_count),
        "energy_time": energy_time,
        "forward_gradient_time": forward_time,
        "reverse_gradient_time": reverse_time,
        "forward_ratio": forward_time / energy_time,
        "reverse_ratio": reverse_time / energy_time,
        "forward_relative_error": relative_error(forward_result, reference_flat),
        "reverse_relative_error": relative_error(reverse_result, reference_flat),
    }


def validate_results(results: list[dict[str, float | int]]) -> None:
    if any(row["forward_relative_error"] >= 1e-12 for row in results):
        raise ValueError("forward relative error threshold failed")
    if any(row["reverse_relative_error"] >= 1e-12 for row in results):
        raise ValueError("reverse relative error threshold failed")
    if results[-1]["forward_ratio"] <= 5 * results[0]["forward_ratio"]:
        raise ValueError("forward ratio did not grow strongly across the tested sizes")
    if results[-1]["forward_ratio"] <= 100 * results[-1]["reverse_ratio"]:
        raise ValueError("P=3072 forward/reverse ratio separation threshold failed")


def plot_scaling(results: list[dict[str, float | int]], output_path: Path) -> None:
    inputs = [row["P"] for row in results]
    forward_ratios = [row["forward_ratio"] for row in results]
    reverse_ratios = [row["reverse_ratio"] for row in results]
    reference = [forward_ratios[0] * value / inputs[0] for value in inputs]

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axis.loglog(inputs, forward_ratios, "o-", label="Forward mode")
    axis.loglog(inputs, reverse_ratios, "s-", label="Reverse mode")
    axis.loglog(inputs, reference, ":", label="Reference proportional to P")
    axis.set_xlabel("Number of scalar inputs P = 3N")
    axis.set_ylabel("Gradient time / energy time")
    axis.set_title("Lennard-Jones cluster AD scaling")
    axis.legend()
    axis.grid(True, which="both", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, format="png")
    plt.close(figure)


def write_evidence(
    results: list[dict[str, float | int]], output_path: Path, repetitions: int
) -> None:
    evidence = {
        "atom_counts": list(ATOM_COUNTS),
        "random_seed": RANDOM_SEED,
        "lattice_spacing": LATTICE_SPACING,
        "perturbation_std": PERTURBATION_STD,
        "periodic_boundary_conditions": False,
        "pair_cutoff": None,
        "repetitions": repetitions,
        "timing_method": f"median of {repetitions} synchronized wall-clock samples after JIT warm-up; all kernels use the same flattened coordinate input",
        "timed_callables": {
            "energy": "jax.jit(flat_energy)(flat_coordinates)",
            "forward": "P calls to jax.jit(jax.jvp(flat_energy, ...))[1]",
            "reverse": "jax.jit(jax.grad(flat_energy))(flat_coordinates)",
        },
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def print_results(results: list[dict[str, float | int]]) -> None:
    print(
        "N     P      energy_s       forward_s      reverse_s      "
        "forward_ratio  reverse_ratio  forward_rel_err  reverse_rel_err"
    )
    for row in results:
        print(
            f"{row['N']:4d} {row['P']:5d} "
            f"{row['energy_time']:.6e} {row['forward_gradient_time']:.6e} "
            f"{row['reverse_gradient_time']:.6e} "
            f"{row['forward_ratio']:.6e} {row['reverse_ratio']:.6e} "
            f"{row['forward_relative_error']:.6e} {row['reverse_relative_error']:.6e}"
        )
    final = results[-1]
    print(f"P=3072 forward ratio: {final['forward_ratio']:.17g}")
    print(f"P=3072 reverse ratio: {final['reverse_ratio']:.17g}")
    print(
        "P=3072 forward/reverse ratio separation: "
        f"{final['forward_ratio'] / final['reverse_ratio']:.17g}"
    )
    print(
        "largest forward relative error: "
        f"{max(row['forward_relative_error'] for row in results):.17g}"
    )
    print(
        "largest reverse relative error: "
        f"{max(row['reverse_relative_error'] for row in results):.17g}"
    )


def run_benchmark(
    repetitions: int = DEFAULT_REPETITIONS,
    output_directory: Path = Path("artifacts/ad"),
) -> list[dict[str, float | int]]:
    results = [benchmark_one(atom_count, repetitions) for atom_count in ATOM_COUNTS]
    validate_results(results)
    plot_scaling(results, output_directory / "scaling.png")
    write_evidence(results, output_directory / "scaling.json", repetitions)
    print_results(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts/ad"))
    args = parser.parse_args()
    run_benchmark(args.repetitions, args.output_directory)


if __name__ == "__main__":
    main()
