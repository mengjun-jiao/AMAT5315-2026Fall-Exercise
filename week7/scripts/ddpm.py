"""1000-step DDPM sampler for the trained Week 7 noise predictor."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diffusion import diffusion_schedule
from noise_predictor import predict_noise


def ddpm_sample(
    parameters: dict[str, tuple[jax.Array, ...]],
    key: jax.Array,
    sample_count: int,
    steps: int = 1000,
) -> jax.Array:
    """Generate a batch with the DDPM reverse update and no final noise."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    betas, alphas, alpha_bars = diffusion_schedule(steps=steps)
    initial_key, noise_key = jax.random.split(key)
    state = jax.random.normal(initial_key, shape=(sample_count, 2), dtype=jnp.float32)
    reverse_keys = jax.random.split(noise_key, steps)
    for reverse_index, t in enumerate(range(steps, 0, -1)):
        beta_t = betas[t - 1]
        alpha_t = alphas[t - 1]
        alpha_bar_t = alpha_bars[t - 1]
        epsilon_hat = predict_noise(parameters, state, jnp.asarray(t, dtype=jnp.float32))
        mean = (state - beta_t / jnp.sqrt(1.0 - alpha_bar_t) * epsilon_hat) / jnp.sqrt(alpha_t)
        if t > 1:
            noise = jax.random.normal(
                reverse_keys[reverse_index], shape=state.shape, dtype=jnp.float32
            )
        else:
            noise = jnp.zeros_like(state)
        state = mean + jnp.sqrt(beta_t) * noise
    return state
