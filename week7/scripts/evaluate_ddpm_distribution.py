"""Evaluate the final DDPM sample distribution against the target mixture."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from ddpm import ddpm_sample
from diffusion import TARGET_CENTERS, TARGET_COMPONENT_STD, sample_target
from smoke_test_ddpm import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "artifacts" / "checkpoints" / "update_025000.npz"
OUTPUT_PATH = ROOT / "artifacts" / "part1_ddpm_distribution.json"
FIGURE_PATH = ROOT / "evidence" / "part1_ddpm_distribution.png"

SEED = 7501
TARGET_SEED = 7502
SAMPLE_COUNT = 2_000
CHECKPOINT_UPDATE = 25000
STEPS = 1000
POPULATION_TOLERANCE = 0.04
CENTER_TOLERANCE = 0.05


def assign_clusters(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def summarize_clusters(samples: np.ndarray, centers: np.ndarray) -> dict:
    assignments = assign_clusters(samples, centers)
    records = []
    for index, center in enumerate(centers):
        cluster = samples[assignments == index]
        if len(cluster) == 0:
            raise AssertionError(f"cluster {index} is empty")
        mean = cluster.mean(axis=0)
        spread = cluster.std(axis=0, ddof=1)
        records.append(
            {
                "target_center": center.tolist(),
                "population": float(len(cluster) / len(samples)),
                "count": int(len(cluster)),
                "centre": mean.tolist(),
                "spread": spread.tolist(),
                "centre_absolute_error": np.abs(mean - center).tolist(),
                "spread_absolute_error": np.abs(spread - TARGET_COMPONENT_STD).tolist(),
            }
        )
    return {
        "clusters": records,
        "sample_mean": samples.mean(axis=0).tolist(),
        "sample_variance": samples.var(axis=0, ddof=1).tolist(),
    }


def save_figure(generated: np.ndarray, target: np.ndarray, centers: np.ndarray) -> None:
    combined = np.concatenate((generated, target))
    bound = float(np.ceil((np.max(np.abs(combined)) + 0.25) * 10.0) / 10.0)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, samples, title in zip(
        axes, (generated, target), ("DDPM update 25000", "Exact target mixture")
    ):
        assignments = assign_clusters(samples, centers)
        for index in range(4):
            cluster = samples[assignments == index]
            axis.scatter(cluster[:, 0], cluster[:, 1], s=5, alpha=0.45, label=f"cluster {index + 1}")
        axis.scatter(centers[:, 0], centers[:, 1], c="black", marker="x", s=80, label="target centers")
        axis.set_xlim(-bound, bound)
        axis.set_ylim(-bound, bound)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.suptitle("Final DDPM distribution and target distribution")
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    centers = np.asarray(TARGET_CENTERS, dtype=np.float32)
    parameters = load_checkpoint(CHECKPOINT_PATH)

    key = jax.random.key(SEED)
    initial_key, reverse_key = jax.random.split(key)
    initial_noise = jax.random.normal(initial_key, (SAMPLE_COUNT, 2), dtype=jax.numpy.float32)
    reverse_keys = jax.random.split(reverse_key, STEPS)
    reverse_noise = jax.vmap(
        lambda noise_key: jax.random.normal(noise_key, (SAMPLE_COUNT, 2), dtype=jax.numpy.float32)
    )(reverse_keys)
    generated = np.asarray(
        ddpm_sample(
            parameters,
            jax.random.key(SEED),
            SAMPLE_COUNT,
            STEPS,
            initial_noise=initial_noise,
            reverse_noise=reverse_noise,
        )
    )
    if generated.shape != (SAMPLE_COUNT, 2) or not np.all(np.isfinite(generated)):
        raise AssertionError("DDPM generated samples have an invalid shape or non-finite values")

    target, _ = sample_target(jax.random.key(TARGET_SEED), SAMPLE_COUNT)
    target = np.asarray(target)
    generated_summary = summarize_clusters(generated, centers)
    target_summary = summarize_clusters(target, centers)
    population_pass = all(
        abs(cluster["population"] - 0.25) <= POPULATION_TOLERANCE
        for cluster in generated_summary["clusters"]
    )
    centre_pass = all(
        max(cluster["centre_absolute_error"]) <= CENTER_TOLERANCE
        for cluster in generated_summary["clusters"]
    )
    output = {
        "seed": SEED,
        "target_seed": TARGET_SEED,
        "sample_count": SAMPLE_COUNT,
        "checkpoint_update": CHECKPOINT_UPDATE,
        "ddpm_steps": STEPS,
        "no_clipping": True,
        "final_step_noise": False,
        "target_centers": centers.tolist(),
        "target_component_standard_deviation": TARGET_COMPONENT_STD,
        "generated": generated_summary,
        "exact_target_control": target_summary,
        "acceptance": {
            "population_tolerance": POPULATION_TOLERANCE,
            "centre_tolerance": CENTER_TOLERANCE,
            "population_pass": population_pass,
            "centre_pass": centre_pass,
            "spread_pass": None,
            "overall_pass": None,
            "spread_status": "Not evaluated: the supplied task text was truncated at '0.30 ± 0'.",
        },
        "runtime_seconds": time.perf_counter() - started,
        "device": str(jax.devices()[0]),
    }
    save_figure(generated, target, centers)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["generated"], indent=2))
    print(json.dumps(output["acceptance"], indent=2))
    print(f"wrote {FIGURE_PATH}")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
