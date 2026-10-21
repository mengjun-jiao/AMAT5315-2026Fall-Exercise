"""RePaint sampler for the prepared CIFAR-10 test images."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cifar_unet import CifarUNet2D
from diffusion import diffusion_schedule


def repaint_sample(
    model: CifarUNet2D,
    observed: jax.Array,
    observed_mask: jax.Array,
    seed: int,
    levels: tuple[int, ...] = tuple(range(1000, -1, -20)),
) -> jax.Array:
    """Run Algorithm 6 with a fixed observed image and binary mask.

    ``observed_mask`` is true where pixels are known. The sampler receives no
    hidden target values; the target is used only by the experiment runner
    after this function returns.
    """
    if observed.shape != (3, 32, 32):
        raise ValueError(f"expected one NCHW image without batch, got {observed.shape}")
    if observed_mask.shape != observed.shape:
        raise ValueError("observed mask must have the same shape as the image")
    if levels[-1] != 0 or levels[0] != 1000 or len(levels) != 51:
        raise ValueError("RePaint levels must be 1000, 980, ..., 20, 0")

    betas, alphas, alpha_bars = diffusion_schedule(1000)
    observed = jnp.asarray(observed, dtype=jnp.float32)
    observed_mask = jnp.asarray(observed_mask, dtype=jnp.bool_)
    key = jax.random.key(seed)
    key, initial_key = jax.random.split(key)
    state = jax.random.normal(initial_key, observed.shape, dtype=jnp.float32)

    @jax.jit
    def reverse_update(current, t, t_prime, reverse_noise):
        t_index = jnp.asarray(t, dtype=jnp.int32) - 1
        tp_index = jnp.asarray(t_prime, dtype=jnp.int32) - 1
        beta_t = betas[t_index]
        alpha_bar_t = alpha_bars[t_index]
        alpha_bar_tp = jnp.where(t_prime == 0, 1.0, alpha_bars[tp_index])
        a = alpha_bar_t / alpha_bar_tp
        eps_hat = model(current[None, ...], jnp.asarray([t], dtype=jnp.float32))[0]
        x0_hat = (current - jnp.sqrt(1.0 - alpha_bar_t) * eps_hat) / jnp.sqrt(alpha_bar_t)
        x0_hat = jnp.clip(x0_hat, -1.0, 1.0)
        mean = (current - (1.0 - a) * eps_hat / jnp.sqrt(1.0 - alpha_bar_t)) / jnp.sqrt(a)
        variance = jnp.where(
            t_prime == 0,
            0.0,
            (1.0 - alpha_bar_tp) / (1.0 - alpha_bar_t) * (1.0 - a),
        )
        proposal = mean + jnp.sqrt(jnp.maximum(variance, 0.0)) * reverse_noise
        return proposal

    for jump_index, (t, t_prime) in enumerate(zip(levels[:-1], levels[1:])):
        attempts = 1 if t_prime == 0 else 2
        for attempt in range(attempts):
            if attempt == 1:
                key, forward_key = jax.random.split(key)
                forward_noise = jax.random.normal(forward_key, observed.shape, dtype=jnp.float32)
                alpha_bar_t = alpha_bars[t - 1]
                alpha_bar_tp = alpha_bars[t_prime - 1]
                a = alpha_bar_t / alpha_bar_tp
                state = jnp.sqrt(a) * state + jnp.sqrt(1.0 - a) * forward_noise

            key, reverse_key = jax.random.split(key)
            reverse_noise = jax.random.normal(reverse_key, observed.shape, dtype=jnp.float32)
            state = reverse_update(
                state,
                jnp.asarray(t, dtype=jnp.float32),
                jnp.asarray(t_prime, dtype=jnp.int32),
                reverse_noise,
            )

            if t_prime == 0:
                observed_at_next = observed
            else:
                key, observation_key = jax.random.split(key)
                observation_noise = jax.random.normal(observation_key, observed.shape, dtype=jnp.float32)
                alpha_bar_next = alpha_bars[t_prime - 1]
                observed_at_next = (
                    jnp.sqrt(alpha_bar_next) * observed
                    + jnp.sqrt(1.0 - alpha_bar_next) * observation_noise
                )
            state = jnp.where(observed_mask, observed_at_next, state)

    return jnp.where(observed_mask, observed, state)
