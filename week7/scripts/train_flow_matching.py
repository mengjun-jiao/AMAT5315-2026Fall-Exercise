"""Train the Week 7 Flow Matching velocity predictor."""

from __future__ import annotations

import json
import math
import time
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from diffusion import sample_target
from flow_model import initialize_parameters, predict_velocity
from train_diffusion import adam_init, adam_update


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "flow.toml"
OUTPUT_PATH = ROOT / "artifacts" / "part3_flow_training.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "checkpoints" / "flow_update_025000.npz"


def load_config() -> dict:
    with CONFIG_PATH.open("rb") as stream:
        return tomllib.load(stream)


def save_checkpoint(parameters: dict[str, tuple[jax.Array, ...]]) -> str:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index, weight in enumerate(parameters["weights"]):
        arrays[f"weight_{index}"] = np.asarray(weight)
    for index, bias in enumerate(parameters["biases"]):
        arrays[f"bias_{index}"] = np.asarray(bias)
    np.savez_compressed(CHECKPOINT_PATH, **arrays)
    return str(CHECKPOINT_PATH.relative_to(ROOT.parent))


def cosine_learning_rate(update: int, total_updates: int, initial: float) -> float:
    fraction = update / total_updates
    return initial * 0.5 * (1.0 + math.cos(math.pi * fraction))


def main() -> None:
    config = load_config()
    model_config = config["model"]
    training_config = config["training"]
    updates = int(training_config["updates"])
    batch_size = int(training_config["batch_size"])
    seed = int(training_config["seed"])
    key = jax.random.key(seed)
    initialization_key, key = jax.random.split(key)
    parameters = initialize_parameters(
        initialization_key,
        hidden_widths=tuple(model_config["hidden_widths"]),
        time_feature_count=int(model_config["time_features"]),
    )
    optimizer_state = adam_init(parameters)

    def loss_function(params, x_zero, noise, times):
        x_s = (1.0 - times[:, None]) * x_zero + times[:, None] * noise
        target_velocity = noise - x_zero
        prediction = predict_velocity(params, x_s, times)
        return jnp.mean(0.5 * jnp.sum(jnp.square(prediction - target_velocity), axis=1))

    @jax.jit
    def train_step(params, state, x_zero, noise, times, learning_rate):
        loss, gradients = jax.value_and_grad(loss_function)(params, x_zero, noise, times)
        params, state = adam_update(
            params,
            gradients,
            state,
            learning_rate,
            float(training_config["beta1"]),
            float(training_config["beta2"]),
            float(training_config["epsilon"]),
        )
        return params, state, loss

    checkpoint_file = save_checkpoint(parameters)
    loss_history: list[float] = []
    started = time.perf_counter()
    for update in range(1, updates + 1):
        key, data_key, noise_key, time_key = jax.random.split(key, 4)
        x_zero, _ = sample_target(data_key, batch_size)
        noise = jax.random.normal(noise_key, (batch_size, 2), dtype=jnp.float32)
        times = jax.random.uniform(time_key, (batch_size,), minval=0.0, maxval=1.0)
        learning_rate = cosine_learning_rate(
            update, updates, float(training_config["learning_rate"])
        )
        parameters, optimizer_state, loss = train_step(
            parameters,
            optimizer_state,
            x_zero,
            noise,
            times,
            jnp.asarray(learning_rate, dtype=jnp.float32),
        )
        loss_value = float(loss)
        loss_history.append(loss_value)
        if update in (1, 100, 1000, 5000, 10000, 25000):
            print(f"update {update:5d}: loss={loss_value:.7f}, lr={learning_rate:.8f}", flush=True)

    checkpoint_file = save_checkpoint(parameters)
    runtime = time.perf_counter() - started
    output = {
        "config": config,
        "seed": seed,
        "loss_history": loss_history,
        "updates": updates,
        "checkpoint": checkpoint_file,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "runtime_seconds": runtime,
        "device": str(jax.devices()[0]),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"updates: {len(loss_history)}")
    print(f"initial loss: {loss_history[0]:.7f}")
    print(f"final loss: {loss_history[-1]:.7f}")
    print(f"runtime: {runtime:.2f} s")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
