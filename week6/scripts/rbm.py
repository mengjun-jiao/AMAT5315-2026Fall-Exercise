"""Train MNIST restricted Boltzmann models with CD-1 and CD-10."""

from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from datasets import load_mnist


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "rbm.toml"
INPUT_PATH = ROOT / "inputs" / "mnist"
OUTPUT_PATH = ROOT / "artifacts" / "rbm_training.json"
PARAMETERS_PATH = ROOT / "artifacts" / "rbm_model_parameters.npz"


def sigmoid_probability(logits: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(logits)


def sample_bernoulli(key: jax.Array, probabilities: jax.Array) -> jax.Array:
    return jax.random.bernoulli(key, probabilities).astype(probabilities.dtype)


def hidden_probability(v: jax.Array, weights: jax.Array, hidden_bias: jax.Array) -> jax.Array:
    """Return p(h=1|v) for a batch of visible states."""
    return sigmoid_probability(v @ weights + hidden_bias)


def visible_probability(v: jax.Array, weights: jax.Array, visible_bias: jax.Array) -> jax.Array:
    """Return p(v=1|h) for a batch of hidden states."""
    return sigmoid_probability(v @ weights.T + visible_bias)


def energy(
    v: jax.Array,
    h: jax.Array,
    weights: jax.Array,
    visible_bias: jax.Array,
    hidden_bias: jax.Array,
) -> jax.Array:
    """Evaluate E(v,h) = -a^T v - b^T h - v^T W h."""
    return (
        -v @ visible_bias
        -h @ hidden_bias
        -jnp.sum((v @ weights) * h, axis=-1)
    )


def free_energy(
    v: jax.Array,
    weights: jax.Array,
    visible_bias: jax.Array,
    hidden_bias: jax.Array,
) -> jax.Array:
    """Evaluate the visible free energy, up to its additive constant."""
    hidden_logits = v @ weights + hidden_bias
    return -v @ visible_bias -jnp.sum(jax.nn.softplus(hidden_logits), axis=-1)


def gibbs_step(
    key: jax.Array,
    v: jax.Array,
    weights: jax.Array,
    visible_bias: jax.Array,
    hidden_bias: jax.Array,
) -> jax.Array:
    """Run one block Gibbs step, h|v followed by v|h."""
    hidden_key, visible_key = jax.random.split(key)
    h = sample_bernoulli(hidden_key, hidden_probability(v, weights, hidden_bias))
    return sample_bernoulli(visible_key, visible_probability(h, weights, visible_bias))


def gibbs_sample(
    key: jax.Array,
    v: jax.Array,
    weights: jax.Array,
    visible_bias: jax.Array,
    hidden_bias: jax.Array,
    steps: int,
) -> jax.Array:
    """Run the requested number of alternating block Gibbs steps."""
    keys = jax.random.split(key, steps)

    def transition(state, step_key):
        next_state = gibbs_step(step_key, state, weights, visible_bias, hidden_bias)
        return next_state, None

    final_state, _ = jax.lax.scan(transition, v, keys)
    return final_state


def make_update(steps: int, learning_rate: float, weight_decay: float):
    """Build a compiled SGD update for a fixed CD chain length."""

    def loss_function(params, v0, key):
        weights, visible_bias, hidden_bias = params
        vk = gibbs_sample(key, v0, weights, visible_bias, hidden_bias, steps)
        positive = free_energy(v0, weights, visible_bias, hidden_bias)
        negative = free_energy(jax.lax.stop_gradient(vk), weights, visible_bias, hidden_bias)
        surrogate = jnp.mean(positive - negative)
        regularization = 0.5 * weight_decay * jnp.sum(weights * weights)
        return surrogate + regularization

    @jax.jit
    def update(params, v0, key):
        loss, gradients = jax.value_and_grad(loss_function)(params, v0, key)
        weights, visible_bias, hidden_bias = params
        grad_weights, grad_visible_bias, grad_hidden_bias = gradients
        new_params = (
            weights - learning_rate * grad_weights,
            visible_bias - learning_rate * grad_visible_bias,
            hidden_bias - learning_rate * grad_hidden_bias,
        )
        return new_params, loss

    return update


def initialize_parameters(
    train_images: np.ndarray,
    visible_units: int,
    hidden_units: int,
    weight_std: float,
    hidden_bias_value: float,
    clip_low: float,
    clip_high: float,
    seed: int,
):
    frequencies = train_images.mean(axis=0)
    clipped = np.clip(frequencies, clip_low, clip_high)
    visible_bias = np.log(clipped / (1.0 - clipped)).astype(np.float32)
    key = jax.random.PRNGKey(seed)
    weights = weight_std * jax.random.normal(key, (visible_units, hidden_units), dtype=jnp.float32)
    return (
        weights,
        jnp.asarray(visible_bias),
        jnp.full((hidden_units,), hidden_bias_value, dtype=jnp.float32),
    )


def reconstruction_error(params, test_images: jax.Array, key: jax.Array) -> float:
    weights, visible_bias, hidden_bias = params
    hidden_key, visible_key = jax.random.split(key)
    hidden = sample_bernoulli(hidden_key, hidden_probability(test_images, weights, hidden_bias))
    reconstruction = sample_bernoulli(visible_key, visible_probability(hidden, weights, visible_bias))
    return float(jnp.mean(reconstruction != test_images))


def evaluate(params, test_images: jax.Array, seed: int, epoch: int) -> float:
    key = jax.random.fold_in(jax.random.PRNGKey(seed), epoch)
    return reconstruction_error(params, test_images, key)


def train_model(
    initial_params,
    train_images: jax.Array,
    test_images: jax.Array,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    cd_steps: int,
    shuffle_seed: int,
    evaluation_seed: int,
) -> dict:
    update = make_update(cd_steps, learning_rate, weight_decay)
    params = initial_params
    batches = train_images.shape[0] // batch_size
    history = [
        {
            "epoch": 0,
            "cd_surrogate": None,
            "held_out_one_step_reconstruction_error": evaluate(
                params, test_images, evaluation_seed, 0
            ),
        }
    ]

    for epoch in range(1, epochs + 1):
        epoch_key = jax.random.fold_in(jax.random.PRNGKey(shuffle_seed), epoch)
        shuffle_key, gibbs_key = jax.random.split(epoch_key)
        order = np.asarray(jax.random.permutation(shuffle_key, train_images.shape[0]))
        losses = []
        for batch_index in range(batches):
            indices = order[batch_index * batch_size : (batch_index + 1) * batch_size]
            batch = train_images[indices]
            batch_key = jax.random.fold_in(gibbs_key, batch_index)
            params, loss = update(params, batch, batch_key)
            losses.append(float(loss))
        history.append(
            {
                "epoch": epoch,
                "cd_surrogate": float(np.mean(losses)),
                "held_out_one_step_reconstruction_error": evaluate(
                    params, test_images, evaluation_seed, epoch
                ),
            }
        )
        print(
            f"CD-{cd_steps} epoch {epoch:02d}/{epochs}: "
            f"surrogate={history[-1]['cd_surrogate']:.6f}, "
            f"reconstruction_error={history[-1]['held_out_one_step_reconstruction_error']:.6f}"
        )
    return {"history": history, "params": params}


def main() -> None:
    started = time.perf_counter()
    with CONFIG_PATH.open("rb") as stream:
        config = tomllib.load(stream)

    train_images, _, test_images, _ = load_mnist(INPUT_PATH)
    train_binary = (train_images.reshape(len(train_images), -1) >= 128).astype(np.float32)
    test_binary = (test_images[:64].reshape(64, -1) >= 128).astype(np.float32)

    training = config["training"]
    model = config["model"]
    initialization = config["initialization"]
    seeds = config["seeds"]
    initial_params = initialize_parameters(
        train_binary,
        model["visible_units"],
        model["hidden_units"],
        initialization["weight_std"],
        initialization["hidden_bias"],
        initialization["pixel_frequency_clip"][0],
        initialization["pixel_frequency_clip"][1],
        seeds["initialization"],
    )
    train_array = jnp.asarray(train_binary)
    test_array = jnp.asarray(test_binary)

    models = {}
    trained_parameters = {}
    for cd_steps in (1, 10):
        result = train_model(
            initial_params,
            train_array,
            test_array,
            training["epochs"],
            training["batch_size"],
            training["learning_rate"],
            training["weight_decay"],
            cd_steps,
            seeds["training"],
            seeds["evaluation"],
        )
        models[f"CD-{cd_steps}"] = {"history": result["history"]}
        trained_parameters[f"cd{cd_steps}_weights"] = np.asarray(result["params"][0])
        trained_parameters[f"cd{cd_steps}_visible_bias"] = np.asarray(result["params"][1])
        trained_parameters[f"cd{cd_steps}_hidden_bias"] = np.asarray(result["params"][2])

    np.savez_compressed(PARAMETERS_PATH, **trained_parameters)

    output = {
        "training_settings": config,
        "preprocessing": {
            "dataset": "MNIST",
            "threshold": 128,
            "visible_shape": [784],
            "evaluation_images": "first 64 test images",
        },
        "models": models,
        "model_parameters": str(PARAMETERS_PATH.relative_to(ROOT)),
        "training_time_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    print(f"training time: {output['training_time_seconds']:.2f} s")


if __name__ == "__main__":
    main()
