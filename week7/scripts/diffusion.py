"""Target distribution sampling and forward diffusion utilities for Week 7."""

from __future__ import annotations

import jax
import jax.numpy as jnp


TARGET_CENTERS = jnp.asarray(
    ((3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -3.0)),
    dtype=jnp.float32,
)
TARGET_COMPONENT_WEIGHTS = jnp.full((4,), 0.25, dtype=jnp.float32)
TARGET_COMPONENT_STD = 0.3
DIFFUSION_STEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02


def sample_target(key: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
    """Sample ``n`` points and their component labels from the four Gaussians."""
    if n < 0:
        raise ValueError("n must be non-negative")
    component_key, noise_key = jax.random.split(key)
    components = jax.random.choice(
        component_key,
        TARGET_CENTERS.shape[0],
        shape=(n,),
        p=TARGET_COMPONENT_WEIGHTS,
    )
    noise = jax.random.normal(noise_key, shape=(n, 2), dtype=jnp.float32)
    samples = TARGET_CENTERS[components] + TARGET_COMPONENT_STD * noise
    return samples, components


def diffusion_schedule(
    steps: int = DIFFUSION_STEPS,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return beta, alpha, and cumulative alpha products for 1-based steps."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < beta_start < 1.0 or not 0.0 < beta_end < 1.0:
        raise ValueError("beta values must be between zero and one")
    betas = jnp.linspace(beta_start, beta_end, steps, dtype=jnp.float32)
    alphas = 1.0 - betas
    alpha_bars = jnp.cumprod(alphas)
    return betas, alphas, alpha_bars


def _schedule_value(values: jax.Array, t: int) -> jax.Array:
    """Select the value for mathematical step t, where t is 1 through T."""
    if not 1 <= t <= values.shape[0]:
        raise ValueError(f"t must be in [1, {values.shape[0]}], got {t}")
    return values[t - 1]


def forward_diffusion_step(
    x_previous: jax.Array,
    key: jax.Array,
    t: int,
    betas: jax.Array,
    alphas: jax.Array | None = None,
) -> jax.Array:
    """Apply one forward step, using the mathematical noise level t."""
    beta_t = _schedule_value(betas, t)
    alpha_t = (1.0 - beta_t) if alphas is None else _schedule_value(alphas, t)
    noise = jax.random.normal(key, shape=x_previous.shape, dtype=x_previous.dtype)
    return jnp.sqrt(alpha_t) * x_previous + jnp.sqrt(beta_t) * noise


def direct_forward_diffusion(
    x_zero: jax.Array,
    key: jax.Array,
    t: int,
    alpha_bars: jax.Array,
) -> jax.Array:
    """Sample x_t directly from x_0 using the cumulative alpha product."""
    alpha_bar_t = _schedule_value(alpha_bars, t)
    noise = jax.random.normal(key, shape=x_zero.shape, dtype=x_zero.dtype)
    return jnp.sqrt(alpha_bar_t) * x_zero + jnp.sqrt(1.0 - alpha_bar_t) * noise
