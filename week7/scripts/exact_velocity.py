"""Exact marginal velocity field for the four-Gaussian Flow Matching path."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diffusion import TARGET_CENTERS


COMPONENT_VARIANCE = 0.09


def exact_marginal_velocity(x_s: jax.Array, s: jax.Array) -> jax.Array:
    """Return the posterior-weighted analytic marginal velocity."""
    x_array = jnp.asarray(x_s, dtype=jnp.float32)
    single = x_array.ndim == 1
    if single:
        x_array = x_array[None, :]
    if x_array.ndim != 2 or x_array.shape[1] != 2:
        raise ValueError(f"expected x_s shape (batch, 2), got {x_array.shape}")
    s_array = jnp.asarray(s, dtype=jnp.float32).reshape(-1)
    if s_array.shape[0] == 1:
        s_array = jnp.broadcast_to(s_array, (x_array.shape[0],))
    if s_array.shape[0] != x_array.shape[0]:
        raise ValueError("time batch and sample batch must have equal length")

    one_minus_s = 1.0 - s_array
    sigma_squared = one_minus_s**2 * COMPONENT_VARIANCE + s_array**2
    component_means = one_minus_s[:, None, None] * TARGET_CENTERS[None, :, :]
    residual = x_array[:, None, :] - component_means
    log_weights = -0.5 * jnp.sum(jnp.square(residual), axis=-1) / sigma_squared[:, None]
    posterior = jax.nn.softmax(log_weights, axis=1)
    coefficient = (s_array - one_minus_s * COMPONENT_VARIANCE) / sigma_squared
    component_velocity = (
        -TARGET_CENTERS[None, :, :]
        + coefficient[:, None, None] * residual
    )
    result = jnp.sum(posterior[:, :, None] * component_velocity, axis=1)
    return result[0] if single else result
