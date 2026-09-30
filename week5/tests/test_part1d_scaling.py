"""Fast validation tests for the Week 5 Part 1D cluster benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import jax.numpy as jnp

import part1d_scaling


def test_pair_counting() -> None:
    pair_i, pair_j = part1d_scaling.pair_indices(4)
    assert len(pair_i) == 6
    assert len(pair_j) == 6
    assert all(int(i) < int(j) for i, j in zip(pair_i, pair_j))


def test_analytic_gradient_sign_and_accumulation() -> None:
    distance = 1.2
    coordinates = jnp.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=jnp.float64)
    gradient = part1d_scaling.analytic_gradient(coordinates)
    radial_derivative = 24 * (distance**-7 - 2 * distance**-13)
    assert float(gradient[0, 0]) == -radial_derivative
    assert float(gradient[1, 0]) == radial_derivative
    assert jnp.allclose(gradient[0, 1:], 0.0)
    assert jnp.allclose(gradient[1, 1:], 0.0)
    assert jnp.allclose(jnp.sum(gradient, axis=0), 0.0)


def test_small_forward_and_reverse_gradients() -> None:
    atom_count = 4
    coordinates = part1d_scaling.make_cluster(atom_count)
    cluster_energy = part1d_scaling.make_cluster_energy(atom_count)
    flat_energy = part1d_scaling.make_flat_energy(cluster_energy, atom_count)
    forward_component = part1d_scaling.make_forward_component(flat_energy)
    reverse_gradient = part1d_scaling.make_reverse_gradient(flat_energy)
    flat_coordinates = coordinates.reshape((-1,))

    forward = part1d_scaling.forward_gradient(flat_coordinates, forward_component)
    reverse = reverse_gradient(flat_coordinates)
    reference = part1d_scaling.analytic_gradient(coordinates).reshape((-1,))
    forward.block_until_ready()
    reverse.block_until_ready()
    assert part1d_scaling.relative_error(forward, reference) < 1e-12
    assert part1d_scaling.relative_error(reverse, reference) < 1e-12
    assert jnp.allclose(forward, reverse, rtol=1e-12, atol=1e-12)


def test_requested_input_counts() -> None:
    assert [part1d_scaling.scalar_input_count(n) for n in part1d_scaling.ATOM_COUNTS] == [
        192,
        384,
        768,
        1536,
        3072,
    ]


if __name__ == "__main__":
    test_pair_counting()
    test_analytic_gradient_sign_and_accumulation()
    test_small_forward_and_reverse_gradients()
    test_requested_input_counts()
    print("Part 1D tests passed")
