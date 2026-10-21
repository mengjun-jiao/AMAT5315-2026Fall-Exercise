"""Exact conditional-mean noise predictor for the four-Gaussian target."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diffusion import TARGET_CENTERS, TARGET_COMPONENT_STD, diffusion_schedule


def exact_conditional_mean_noise(
    x_t: jax.Array,
    t: int,
    alpha_bars: jax.Array | None = None,
) -> jax.Array:
    """Return E[epsilon | x_t] under the exact Gaussian-mixture model."""
    if not 1 <= t <= 1000:
        raise ValueError("t must be in [1, 1000]")
    x_array = jnp.asarray(x_t, dtype=jnp.float32)
    single = x_array.ndim == 1
    if single:
        x_array = x_array[None, :]
    if x_array.ndim != 2 or x_array.shape[1] != 2:
        raise ValueError(f"expected x_t shape (batch, 2), got {x_array.shape}")
    if alpha_bars is None:
        _, _, alpha_bars = diffusion_schedule()
    alpha_bar_t = alpha_bars[t - 1]
    signal_scale = jnp.sqrt(alpha_bar_t)
    noise_variance = 1.0 - alpha_bar_t
    component_variance = (
        alpha_bar_t * TARGET_COMPONENT_STD**2 + noise_variance
    )
    residual = x_array[:, None, :] - signal_scale * TARGET_CENTERS[None, :, :]
    log_likelihood = -0.5 * jnp.sum(jnp.square(residual), axis=-1) / component_variance
    log_likelihood = log_likelihood - jnp.log(component_variance)
    posterior_weights = jax.nn.softmax(log_likelihood, axis=1)

    posterior_x0_mean = TARGET_CENTERS[None, :, :] + (
        signal_scale * TARGET_COMPONENT_STD**2 / component_variance
    ) * residual
    conditional_noise = (
        x_array[:, None, :] - signal_scale * posterior_x0_mean
    ) / jnp.sqrt(noise_variance)
    result = jnp.sum(posterior_weights[:, :, None] * conditional_noise, axis=1)
    return result[0] if single else result
