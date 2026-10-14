"""Run clamped MNIST image completion and evaluate ink overlap."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

from datasets import load_mnist
from rbm import PARAMETERS_PATH, gibbs_step


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "inputs" / "mnist"
EVIDENCE_PATH = ROOT / "evidence"
OVERLAP_PATH = ROOT / "artifacts" / "rbm_overlap.json"
STEPS = (1, 10, 100, 1000)
CHAIN_COUNT = 8
IMAGE_COUNT = 64
COMPLETION_SEED = 6607


@jax.jit
def advance_clamped(params, state, clamped_upper, keys):
    def transition(current, key):
        next_state = gibbs_step(key, current, *params)
        next_state = next_state.at[:, :, :392].set(clamped_upper[:, None, :])
        return next_state, None

    final_state, _ = jax.lax.scan(transition, state, keys)
    return final_state


def load_parameters(name: str):
    with np.load(PARAMETERS_PATH) as data:
        return (
            jnp.asarray(data[f"{name}_weights"]),
            jnp.asarray(data[f"{name}_visible_bias"]),
            jnp.asarray(data[f"{name}_hidden_bias"]),
        )


def clamped_snapshots(params, initial: jax.Array, clamped_upper: jax.Array, seed: int):
    """Advance chains while replacing the upper half after every Gibbs step."""
    state = initial
    snapshots = []
    previous = 0
    for target in STEPS:
        key = jax.random.fold_in(jax.random.PRNGKey(seed), target)
        keys = jax.random.split(key, target - previous)
        state = advance_clamped(params, state, clamped_upper, keys)
        snapshots.append(np.asarray(state))
        previous = target
    return snapshots


def make_initial_states(correct_upper: np.ndarray, seed: int) -> jax.Array:
    key = jax.random.PRNGKey(seed)
    missing_key = jax.random.fold_in(key, 1)
    missing = jax.random.bernoulli(
        missing_key, 0.5, (IMAGE_COUNT, CHAIN_COUNT, 392)
    ).astype(jnp.float32)
    upper = jnp.asarray(correct_upper.reshape(IMAGE_COUNT, 392))
    upper = jnp.broadcast_to(upper[:, None, :], (IMAGE_COUNT, CHAIN_COUNT, 392))
    return jnp.concatenate((upper, missing), axis=2)


def overlap(completed: np.ndarray, truth: np.ndarray) -> float:
    predicted_ink = completed.reshape(IMAGE_COUNT, CHAIN_COUNT, 28, 28)[:, :, 14:, :].astype(bool)
    truth_ink = truth[:, None, 14:, :].astype(bool)
    intersection = np.logical_and(predicted_ink, truth_ink).sum(axis=(-2, -1))
    predicted_count = predicted_ink.sum(axis=(-2, -1))
    truth_count = truth_ink.sum(axis=(-2, -1))
    denominator = predicted_count + truth_count
    scores = np.divide(
        2.0 * intersection,
        denominator,
        out=np.ones_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )
    return float(scores.mean())


def choose_wrong_upper(labels: np.ndarray, images: np.ndarray) -> np.ndarray:
    wrong = np.empty((len(images), 14, 28), dtype=images.dtype)
    for index, label in enumerate(labels):
        candidates = np.flatnonzero(labels != label)
        wrong[index] = images[candidates[0], :14, :]
    return wrong.reshape(len(images), 392)


def plot_completion(
    samples_by_model: dict[str, list[np.ndarray]], initial: np.ndarray, truth: np.ndarray
) -> None:
    display_count = 8
    model_names = tuple(samples_by_model)
    figure, axes = plt.subplots(
        display_count * len(model_names), 6,
        figsize=(10, 14 * len(model_names)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    columns = ("truth", "masked", "1 step", "10 steps", "100 steps", "1000 steps")
    for model_index, model_name in enumerate(model_names):
        samples = samples_by_model[model_name]
        for row in range(display_count):
            figure_row = model_index * display_count + row
            images = [
                truth[row],
                initial[row, 0].reshape(28, 28),
                *[sample[row, 0].reshape(28, 28) for sample in samples],
            ]
            for column, image in enumerate(images):
                axis = axes[figure_row, column]
                axis.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="none")
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(columns[column])
            if row == 0:
                axes[figure_row, 0].set_ylabel(model_name)
    figure.suptitle("RBM completion with the upper half clamped")
    figure.savefig(EVIDENCE_PATH / "rbm_completion.png", dpi=160)
    plt.close(figure)


def main() -> None:
    _, _, test_images, test_labels = load_mnist(INPUT_PATH)
    truth = (test_images[:IMAGE_COUNT] >= 128).astype(np.float32)
    correct_upper = truth[:, :14, :].reshape(IMAGE_COUNT, 392)
    wrong_upper = choose_wrong_upper(test_labels[:IMAGE_COUNT], truth)
    correct_initial = make_initial_states(correct_upper, COMPLETION_SEED)
    wrong_initial = correct_initial.at[:, :, :392].set(jnp.asarray(wrong_upper[:, None, :]))

    overlap_results = {"steps": list(STEPS), "chains_per_image": CHAIN_COUNT, "models": {}}
    samples_for_plot = {}
    for name in ("CD-1", "CD-10"):
        params = load_parameters(name.lower().replace("-", ""))
        correct_samples = clamped_snapshots(params, correct_initial, correct_upper, COMPLETION_SEED)
        wrong_samples = clamped_snapshots(params, wrong_initial, wrong_upper, COMPLETION_SEED)
        overlap_results["models"][name] = {
            "correct_upper": [overlap(sample, truth) for sample in correct_samples],
            "wrong_upper": [overlap(sample, truth) for sample in wrong_samples],
        }
        samples_for_plot[name] = correct_samples

    plot_completion(samples_for_plot, np.asarray(correct_initial), truth)

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name, color in (("CD-1", "tab:blue"), ("CD-10", "tab:orange")):
        values = overlap_results["models"][name]
        axis.plot(STEPS, values["correct_upper"], marker="o", color=color, label=f"{name}, correct upper")
        axis.plot(STEPS, values["wrong_upper"], marker="x", linestyle="--", color=color, label=f"{name}, wrong upper")
    axis.set_xscale("log")
    axis.set_xticks(STEPS, [str(step) for step in STEPS])
    axis.set_xlabel("Gibbs steps")
    axis.set_ylabel("ink overlap")
    axis.set_title("MNIST completion ink overlap")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(EVIDENCE_PATH / "rbm_overlap.png", dpi=160)
    plt.close(figure)

    overlap_results["protocol"] = {
        "images": IMAGE_COUNT,
        "upper_half_rows": 14,
        "missing_half_rows": 14,
        "metric": "2 * intersection(A, B) / (|A| + |B|)",
        "completion_seed": COMPLETION_SEED,
        "upper_half_clamped": True,
    }
    OVERLAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERLAP_PATH.write_text(json.dumps(overlap_results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {EVIDENCE_PATH / 'rbm_completion.png'}")
    print(f"wrote {EVIDENCE_PATH / 'rbm_overlap.png'}")
    print(f"wrote {OVERLAP_PATH}")


if __name__ == "__main__":
    main()
