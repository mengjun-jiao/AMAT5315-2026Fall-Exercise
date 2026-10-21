"""Deterministic DDIM sampling with the Part 1 noise schedule."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from diffusion import diffusion_schedule
from noise_predictor import predict_noise


def validate_levels(levels: Sequence[int]) -> tuple[int, ...]:
    levels = tuple(int(level) for level in levels)
    if len(levels) < 2 or levels[0] != 1000 or levels[-1] != 0:
        raise ValueError("levels must start at 1000 and end at 0")
    if any(current <= following for current, following in zip(levels, levels[1:])):
        raise ValueError("levels must be strictly decreasing")
    if any(level < 0 or level > 1000 for level in levels):
        raise ValueError("levels must be in [0, 1000]")
    return levels


def alpha_bar_at(level: int, alpha_bars: jax.Array) -> jax.Array:
    if level == 0:
        return jnp.asarray(1.0, dtype=jnp.float32)
    return alpha_bars[level - 1]


def ddim_trajectory(
    parameters: dict[str, tuple[jax.Array, ...]],
    initial_noise: jax.Array,
    levels: Sequence[int],
) -> jax.Array:
    """Return all deterministic DDIM states, including the initial x_1000."""
    levels = validate_levels(levels)
    state = jnp.asarray(initial_noise, dtype=jnp.float32)
    single = state.ndim == 1
    if single:
        state = state[None, :]
    if state.ndim != 2 or state.shape[1] != 2:
        raise ValueError(f"expected initial_noise shape (batch, 2), got {state.shape}")
    _, _, alpha_bars = diffusion_schedule()
    states = [state]
    for t, next_t in zip(levels, levels[1:]):
        alpha_bar_t = alpha_bar_at(t, alpha_bars)
        alpha_bar_next = alpha_bar_at(next_t, alpha_bars)
        epsilon_hat = predict_noise(parameters, state, jnp.asarray(t, dtype=jnp.float32))
        x_zero_hat = (
            state - jnp.sqrt(1.0 - alpha_bar_t) * epsilon_hat
        ) / jnp.sqrt(alpha_bar_t)
        state = (
            jnp.sqrt(alpha_bar_next) * x_zero_hat
            + jnp.sqrt(1.0 - alpha_bar_next) * epsilon_hat
        )
        states.append(state)
    result = jnp.stack(states)
    return result[:, 0, :] if single else result


def ddim_sample(
    parameters: dict[str, tuple[jax.Array, ...]],
    key: jax.Array,
    sample_count: int,
    levels: Sequence[int],
    initial_noise: jax.Array | None = None,
) -> jax.Array:
    """Generate a deterministic DDIM sample batch from one initial noise draw."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    validate_levels(levels)
    single = False
    if initial_noise is None:
        initial_noise = jax.random.normal(key, (sample_count, 2), dtype=jnp.float32)
    else:
        initial_noise = jnp.asarray(initial_noise, dtype=jnp.float32)
        if initial_noise.ndim == 1:
            if sample_count != 1:
                raise ValueError("a single initial point requires sample_count=1")
            single = True
        elif initial_noise.shape != (sample_count, 2):
            raise ValueError(
                f"initial_noise must have shape {(sample_count, 2)}, got {initial_noise.shape}"
            )
    trajectory = ddim_trajectory(parameters, initial_noise, levels)
    return trajectory[-1] if not single else trajectory[-1]
