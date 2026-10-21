"""Time-conditioned velocity network for Flow Matching."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp


INPUT_WIDTH = 2
TIME_FEATURES = 32
TIME_FREQUENCY_BASE = 10000.0
TIME_SCALE = 1000.0
HIDDEN_WIDTHS = (128, 128, 128)


def time_features(
    s: jax.Array,
    feature_count: int = TIME_FEATURES,
    frequency_base: float = TIME_FREQUENCY_BASE,
    time_scale: float = TIME_SCALE,
) -> jax.Array:
    """Return 16 sine and 16 cosine features for continuous time s."""
    if feature_count % 2:
        raise ValueError("feature_count must be even")
    s_array = jnp.asarray(s, dtype=jnp.float32).reshape(-1)
    half = feature_count // 2
    indices = jnp.arange(half, dtype=jnp.float32)
    frequencies = frequency_base ** (-indices / half)
    phase = (time_scale * s_array)[:, None] * frequencies[None, :]
    return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=1)


def initialize_parameters(
    key: jax.Array,
    hidden_widths: Sequence[int] = HIDDEN_WIDTHS,
    time_feature_count: int = TIME_FEATURES,
) -> dict[str, tuple[jax.Array, ...]]:
    """Initialize weights with std 1/sqrt(input width) and zero biases."""
    widths = (INPUT_WIDTH + time_feature_count, *tuple(hidden_widths), 2)
    keys = jax.random.split(key, len(widths) - 1)
    weights = tuple(
        jax.random.normal(subkey, (input_width, output_width), dtype=jnp.float32)
        / jnp.sqrt(jnp.asarray(input_width, dtype=jnp.float32))
        for subkey, input_width, output_width in zip(keys, widths[:-1], widths[1:])
    )
    biases = tuple(
        jnp.zeros((output_width,), dtype=jnp.float32) for output_width in widths[1:]
    )
    return {"weights": weights, "biases": biases}


def predict_velocity(
    parameters: dict[str, tuple[jax.Array, ...]],
    x_s: jax.Array,
    s: jax.Array,
) -> jax.Array:
    """Predict the conditional velocity for one point or a batch."""
    x_array = jnp.asarray(x_s, dtype=jnp.float32)
    single = x_array.ndim == 1
    if single:
        x_array = x_array[None, :]
    if x_array.ndim != 2 or x_array.shape[1] != INPUT_WIDTH:
        raise ValueError(f"expected x_s shape (batch, 2), got {x_array.shape}")
    s_array = jnp.asarray(s, dtype=jnp.float32).reshape(-1)
    if s_array.shape[0] == 1:
        s_array = jnp.broadcast_to(s_array, (x_array.shape[0],))
    if s_array.shape[0] != x_array.shape[0]:
        raise ValueError("time batch and sample batch must have equal length")
    hidden = jnp.concatenate((x_array, time_features(s_array)), axis=1)
    for index, (weight, bias) in enumerate(
        zip(parameters["weights"], parameters["biases"])
    ):
        hidden = hidden @ weight + bias
        if index < len(parameters["weights"]) - 1:
            hidden = jax.nn.silu(hidden)
    return hidden[0] if single else hidden
