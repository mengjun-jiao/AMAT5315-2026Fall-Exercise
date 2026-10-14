"""Generate MNIST samples from the trained CD-1 and CD-10 RBMs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

from rbm import PARAMETERS_PATH, gibbs_sample


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "evidence" / "rbm_generation.png"
STEPS = (1, 10, 100, 1000)
GENERATION_SEED = 6606
SAMPLE_COUNT = 8


def load_parameters(name: str):
    with np.load(PARAMETERS_PATH) as data:
        return (
            jnp.asarray(data[f"{name}_weights"]),
            jnp.asarray(data[f"{name}_visible_bias"]),
            jnp.asarray(data[f"{name}_hidden_bias"]),
        )


def snapshots(params, initial: jax.Array, seed: int) -> list[np.ndarray]:
    state = initial
    snapshots = [np.asarray(state)]
    previous = 0
    for target in STEPS:
        state = gibbs_sample(
            jax.random.fold_in(jax.random.PRNGKey(seed), target),
            state,
            *params,
            target - previous,
        )
        snapshots.append(np.asarray(state))
        previous = target
    return snapshots


def main() -> None:
    initial = jax.random.bernoulli(
        jax.random.PRNGKey(GENERATION_SEED), 0.5, (SAMPLE_COUNT, 784)
    ).astype(jnp.float32)
    cd1 = snapshots(load_parameters("cd1"), initial, GENERATION_SEED)
    cd10 = snapshots(load_parameters("cd10"), initial, GENERATION_SEED)

    figure, axes = plt.subplots(2, 5, figsize=(10, 4.5), constrained_layout=True)
    row_names = ("CD-1", "CD-10")
    column_names = ("initial", "1 step", "10 steps", "100 steps", "1000 steps")
    for row, (name, samples) in enumerate(zip(row_names, (cd1, cd10))):
        for column, batch in enumerate(samples):
            axis = axes[row, column]
            axis.imshow(batch[0].reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation="none")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(column_names[column])
            if column == 0:
                axis.set_ylabel(name)
    figure.suptitle("RBM generation from random binary configurations")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=160)
    plt.close(figure)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
