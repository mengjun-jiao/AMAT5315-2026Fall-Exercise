"""1000-step DDPM sampler for the trained Week 7 noise predictor."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diffusion import diffusion_schedule
from noise_predictor import predict_noise


def _validate_noise(
    initial_noise: jax.Array,
    reverse_noise: jax.Array,
    sample_count: int,
    steps: int,
) -> tuple[jax.Array, jax.Array]:
    initial_noise = jnp.asarray(initial_noise, dtype=jnp.float32)
    reverse_noise = jnp.asarray(reverse_noise, dtype=jnp.float32)
    if initial_noise.shape != (sample_count, 2):
        raise ValueError(
            f"initial_noise must have shape {(sample_count, 2)}, got {initial_noise.shape}"
        )
    if reverse_noise.shape != (steps, sample_count, 2):
        raise ValueError(
            "reverse_noise must have shape "
            f"{(steps, sample_count, 2)}, got {reverse_noise.shape}"
        )
    return initial_noise, reverse_noise


def ddpm_trajectory(
    parameters: dict[str, tuple[jax.Array, ...]],
    initial_noise: jax.Array,
    reverse_noise: jax.Array,
    steps: int = 1000,
) -> jax.Array:
    """Return the complete reverse trajectory, starting at x_T."""
    sample_count = initial_noise.shape[0]
    initial_noise, reverse_noise = _validate_noise(
        initial_noise, reverse_noise, sample_count, steps
    )
    betas, alphas, alpha_bars = diffusion_schedule(steps=steps)
    state = initial_noise
    trajectory = [state]
    for reverse_index, t in enumerate(range(steps, 0, -1)):
        beta_t = betas[t - 1]
        alpha_t = alphas[t - 1]
        alpha_bar_t = alpha_bars[t - 1]
        epsilon_hat = predict_noise(parameters, state, jnp.asarray(t, dtype=jnp.float32))
        mean = (state - beta_t / jnp.sqrt(1.0 - alpha_bar_t) * epsilon_hat) / jnp.sqrt(alpha_t)
        if t > 1:
            noise = reverse_noise[reverse_index]
        else:
            noise = jnp.zeros_like(state)
        state = mean + jnp.sqrt(beta_t) * noise
        trajectory.append(state)
    return jnp.stack(trajectory)


def ddpm_sample(
    parameters: dict[str, tuple[jax.Array, ...]],
    key: jax.Array,
    sample_count: int,
    steps: int = 1000,
    initial_noise: jax.Array | None = None,
    reverse_noise: jax.Array | None = None,
) -> jax.Array:
    """Generate a batch, optionally reusing explicit DDPM noise arrays."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if initial_noise is None or reverse_noise is None:
        initial_key, noise_key = jax.random.split(key)
        if initial_noise is None:
            initial_noise = jax.random.normal(
                initial_key, shape=(sample_count, 2), dtype=jnp.float32
            )
        if reverse_noise is None:
            reverse_keys = jax.random.split(noise_key, steps)
            reverse_noise = jax.vmap(
                lambda reverse_key: jax.random.normal(
                    reverse_key, shape=(sample_count, 2), dtype=jnp.float32
                )
            )(reverse_keys)
    trajectory = ddpm_trajectory(parameters, initial_noise, reverse_noise, steps)
    return trajectory[-1]
