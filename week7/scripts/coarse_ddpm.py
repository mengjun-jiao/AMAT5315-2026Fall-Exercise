"""Coarse-grid DDPM sampling for arbitrary integer noise-level grids."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from diffusion import diffusion_schedule
from noise_predictor import predict_noise


def integer_levels(steps: int) -> tuple[int, ...]:
    if steps <= 0 or steps > 1000:
        raise ValueError("steps must be in [1, 1000]")
    levels = tuple(int(round(value)) for value in jnp.linspace(1000.0, 0.0, steps + 1))
    if len(levels) != steps + 1 or levels[0] != 1000 or levels[-1] != 0:
        raise ValueError("invalid integer level grid")
    if any(current <= following for current, following in zip(levels, levels[1:])):
        raise ValueError(f"cannot construct strictly decreasing grid for {steps} steps")
    return levels


def coarse_ddpm_trajectory(
    parameters: dict[str, tuple[jax.Array, ...]],
    initial_noise: jax.Array,
    reverse_noise: jax.Array,
    levels: Sequence[int],
) -> jax.Array:
    levels = tuple(int(level) for level in levels)
    if len(levels) < 2 or levels[0] != 1000 or levels[-1] != 0:
        raise ValueError("levels must start at 1000 and end at 0")
    if any(current <= following for current, following in zip(levels, levels[1:])):
        raise ValueError("levels must be strictly decreasing")
    state = jnp.asarray(initial_noise, dtype=jnp.float32)
    if state.ndim != 2 or state.shape[1] != 2:
        raise ValueError(f"expected initial_noise shape (batch, 2), got {state.shape}")
    reverse_noise = jnp.asarray(reverse_noise, dtype=jnp.float32)
    expected = (len(levels) - 1, state.shape[0], 2)
    if reverse_noise.shape != expected:
        raise ValueError(f"reverse_noise must have shape {expected}, got {reverse_noise.shape}")
    _, _, alpha_bars = diffusion_schedule()

    def alpha_bar(level: int) -> jax.Array:
        return jnp.asarray(1.0, dtype=jnp.float32) if level == 0 else alpha_bars[level - 1]

    states = [state]
    for index, (current, following) in enumerate(zip(levels, levels[1:])):
        alpha_current = alpha_bar(current)
        alpha_following = alpha_bar(following)
        ratio = alpha_current / alpha_following
        epsilon_hat = predict_noise(
            parameters, state, jnp.asarray(current, dtype=jnp.float32)
        )
        mean = (
            state - (1.0 - ratio) * epsilon_hat / jnp.sqrt(1.0 - alpha_current)
        ) / jnp.sqrt(ratio)
        noise = jnp.zeros_like(state) if following == 0 else reverse_noise[index]
        state = mean + jnp.sqrt(1.0 - ratio) * noise
        states.append(state)
    return jnp.stack(states)


def coarse_ddpm_sample(
    parameters: dict[str, tuple[jax.Array, ...]],
    initial_noise: jax.Array,
    reverse_noise: jax.Array,
    levels: Sequence[int],
) -> jax.Array:
    return coarse_ddpm_trajectory(parameters, initial_noise, reverse_noise, levels)[-1]
