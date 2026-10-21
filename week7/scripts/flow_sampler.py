"""Euler sampler for the trained Flow Matching velocity field."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from flow_model import predict_velocity


def validate_time_grid(times: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(time) for time in times)
    if len(values) < 2 or values[0] != 1.0 or values[-1] != 0.0:
        raise ValueError("time grid must start at 1 and end at 0")
    if any(current <= following for current, following in zip(values, values[1:])):
        raise ValueError("time grid must be strictly decreasing")
    if any(time < 0.0 or time > 1.0 for time in values):
        raise ValueError("time grid values must be in [0, 1]")
    return values


def flow_trajectory(
    parameters: dict[str, tuple[jax.Array, ...]],
    initial_noise: jax.Array,
    times: Sequence[float],
) -> jax.Array:
    """Return all Euler states, including the initial state at s=1."""
    times = validate_time_grid(times)
    state = jnp.asarray(initial_noise, dtype=jnp.float32)
    single = state.ndim == 1
    if single:
        state = state[None, :]
    if state.ndim != 2 or state.shape[1] != 2:
        raise ValueError(f"expected initial_noise shape (batch, 2), got {state.shape}")
    states = [state]
    for current, following in zip(times, times[1:]):
        velocity = predict_velocity(parameters, state, jnp.asarray(current, dtype=jnp.float32))
        state = state - (current - following) * velocity
        states.append(state)
    result = jnp.stack(states)
    return result[:, 0, :] if single else result


def flow_sample(
    parameters: dict[str, tuple[jax.Array, ...]],
    initial_noise: jax.Array,
    times: Sequence[float],
) -> jax.Array:
    return flow_trajectory(parameters, initial_noise, times)[-1]
