"""Train the Week 7 time-conditioned noise predictor."""

from __future__ import annotations

import json
import math
import time
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from diffusion import diffusion_schedule, sample_target
from noise_predictor import initialize_parameters, predict_noise


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "diffusion.toml"
OUTPUT_PATH = ROOT / "artifacts" / "part1_diffusion_training.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "checkpoints"


def load_config() -> dict:
    with CONFIG_PATH.open("rb") as stream:
        return tomllib.load(stream)


def save_checkpoint(
    parameters: dict[str, tuple[jax.Array, ...]], update: int
) -> str:
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    filename = f"update_{update:06d}.npz"
    arrays = {}
    for index, weight in enumerate(parameters["weights"]):
        arrays[f"weight_{index}"] = np.asarray(weight)
    for index, bias in enumerate(parameters["biases"]):
        arrays[f"bias_{index}"] = np.asarray(bias)
    np.savez_compressed(CHECKPOINT_PATH / filename, **arrays)
    return str(Path("artifacts") / "checkpoints" / filename)


def adam_init(parameters: dict[str, tuple[jax.Array, ...]]) -> dict:
    return {
        "step": jnp.asarray(0, dtype=jnp.int32),
        "m": jax.tree_util.tree_map(jnp.zeros_like, parameters),
        "v": jax.tree_util.tree_map(jnp.zeros_like, parameters),
    }


def adam_update(
    parameters: dict,
    gradients: dict,
    state: dict,
    learning_rate: jax.Array,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> tuple[dict, dict]:
    step = state["step"] + 1
    m = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        state["m"],
        gradients,
    )
    v = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * jnp.square(grad),
        state["v"],
        gradients,
    )
    correction_m = 1.0 - beta1**step
    correction_v = 1.0 - beta2**step
    parameters = jax.tree_util.tree_map(
        lambda parameter, first, second: parameter
        - learning_rate * (first / correction_m)
        / (jnp.sqrt(second / correction_v) + epsilon),
        parameters,
        m,
        v,
    )
    return parameters, {"step": step, "m": m, "v": v}


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
    checkpoint_updates = tuple(int(update) for update in training_config["checkpoint_updates"])
    betas, _, alpha_bars = diffusion_schedule()

    key = jax.random.key(seed)
    initialization_key, key = jax.random.split(key)
    parameters = initialize_parameters(
        initialization_key,
        hidden_widths=tuple(model_config["hidden_widths"]),
        time_feature_count=int(model_config["time_features"]),
    )
    optimizer_state = adam_init(parameters)

    def loss_function(params, x_zero, noise, times):
        alpha_bar = alpha_bars[times - 1]
        x_t = (
            jnp.sqrt(alpha_bar)[:, None] * x_zero
            + jnp.sqrt(1.0 - alpha_bar)[:, None] * noise
        )
        prediction = predict_noise(params, x_t, times)
        return jnp.mean(0.5 * jnp.sum(jnp.square(prediction - noise), axis=1))

    @jax.jit
    def train_step(params, optimizer_state, x_zero, noise, times, learning_rate):
        loss, gradients = jax.value_and_grad(loss_function)(params, x_zero, noise, times)
        params, optimizer_state = adam_update(
            params,
            gradients,
            optimizer_state,
            learning_rate,
            float(training_config["beta1"]),
            float(training_config["beta2"]),
            float(training_config["epsilon"]),
        )
        return params, optimizer_state, loss

    checkpoint_files = {0: save_checkpoint(parameters, 0)}
    loss_history: list[float] = []
    started = time.perf_counter()
    for update in range(1, updates + 1):
        key, batch_key, noise_key, time_key = jax.random.split(key, 4)
        x_zero, _ = sample_target(batch_key, batch_size)
        noise = jax.random.normal(noise_key, shape=(batch_size, 2), dtype=jnp.float32)
        times = jax.random.randint(time_key, shape=(batch_size,), minval=1, maxval=1001)
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
        if update in checkpoint_updates:
            checkpoint_files[update] = save_checkpoint(parameters, update)
            print(f"update {update:5d}: loss={loss_value:.7f}, lr={learning_rate:.8f}", flush=True)

    runtime = time.perf_counter() - started
    output = {
        "config": config,
        "seed": seed,
        "loss_history": loss_history,
        "checkpoint_updates": list(checkpoint_updates),
        "checkpoint_files": {str(update): path for update, path in checkpoint_files.items()},
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
