"""Compare Flow Matching, DDIM, and coarse-grid DDPM sampling."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from coarse_ddpm import coarse_ddpm_sample, coarse_ddpm_trajectory, integer_levels
from diffusion import TARGET_CENTERS, sample_target
from ddim import ddim_sample, ddim_trajectory
from flow_model import predict_velocity
from flow_sampler import flow_sample, flow_trajectory
from smoke_test_ddpm import load_checkpoint as load_diffusion_checkpoint
from verify_flow_velocity import load_checkpoint as load_flow_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "update_025000.npz"
FLOW_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "flow_update_025000.npz"
OUTPUT_PATH = ROOT / "artifacts" / "part3_sampling_comparison.json"
FIGURE_PATH = ROOT / "evidence" / "part3_sampling_comparison.png"
DISTRIBUTION_PATH = ROOT / "evidence" / "part3_sampling_distributions.png"

SEED = 9317
SAMPLE_COUNT = 4000
STEP_COUNTS = (1, 2, 5, 10, 20, 50, 100)
DDPM_DDIM_STEPS = (*STEP_COUNTS, 1000)
TARGET_CENTRES = np.asarray(TARGET_CENTERS, dtype=np.float32)


def reverse_noise(key: jax.Array, intervals: int) -> jax.Array:
    return jax.random.normal(
        key, (intervals, SAMPLE_COUNT, 2), dtype=jnp.float32
    )


def distribution_stats(samples: np.ndarray) -> dict:
    distances = np.sum(
        (samples[:, None, :] - TARGET_CENTRES[None, :, :]) ** 2, axis=2
    )
    labels = np.argmin(distances, axis=1)
    clusters = []
    for index, centre in enumerate(TARGET_CENTRES):
        cluster = samples[labels == index]
        empirical_centre = cluster.mean(axis=0)
        spread = cluster.std(axis=0, ddof=1)
        clusters.append(
            {
                "population": float(len(cluster) / len(samples)),
                "centre": empirical_centre.tolist(),
                "centre_error": float(np.linalg.norm(empirical_centre - centre)),
                "spread": spread.tolist(),
            }
        )
    max_population_error = max(abs(cluster["population"] - 0.25) for cluster in clusters)
    max_width_error = max(
        abs(value - 0.30) for cluster in clusters for value in cluster["spread"]
    )
    max_centre_error = max(cluster["centre_error"] for cluster in clusters)
    return {
        "clusters": clusters,
        "max_population_error": float(max_population_error),
        "max_centre_error": float(max_centre_error),
        "max_width_error": float(max_width_error),
        "mean": samples.mean(axis=0).tolist(),
        "variance": samples.var(axis=0, ddof=1).tolist(),
    }


def mean_pairwise_distance(left: np.ndarray, right: np.ndarray, block: int = 256) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(left), block):
        chunk = left[start : start + block]
        distances = np.linalg.norm(chunk[:, None, :] - right[None, :, :], axis=2)
        total += float(distances.sum())
        count += distances.size
    return total / count


def energy_distance(samples: np.ndarray, reference: np.ndarray, reference_self: float) -> float:
    cross = mean_pairwise_distance(samples, reference)
    self_distance = mean_pairwise_distance(samples, samples)
    return float(2.0 * cross - self_distance - reference_self)


def adjacent_consistency_check(
    parameters: dict,
    initial_noise: jax.Array,
    reverse_noise: jax.Array,
) -> dict:
    from ddpm import ddpm_trajectory

    levels = (1000, 999, 0)
    coarse = np.asarray(
        coarse_ddpm_trajectory(parameters, initial_noise, reverse_noise[:2], levels)
    )
    adjacent = np.asarray(ddpm_trajectory(parameters, initial_noise, reverse_noise, steps=1000))
    # The first coarse transition is exactly the first adjacent transition.
    difference = float(np.max(np.abs(coarse[1] - adjacent[1])))
    return {"levels": list(levels), "max_absolute_difference_first_transition": difference}


def plot_comparison(results: dict) -> None:
    methods = ("flow", "ddim", "ddpm")
    labels = {"flow": "Flow Matching", "ddim": "DDIM", "ddpm": "DDPM"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    metrics = (
        ("max_population_error", "Maximum population error"),
        ("max_centre_error", "Maximum centre error"),
        ("max_width_error", "Maximum width error"),
        ("energy_distance", "Energy distance"),
    )
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for method in methods:
            entries = results[method]
            x = [entry["steps"] for entry in entries]
            y = [entry[metric] for entry in entries]
            axis.plot(x, y, marker="o", label=labels[method])
        axis.set_xscale("log")
        axis.set_xlabel("Sampling steps")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Week 7 Part 3 sampling comparison")
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)


def plot_distributions(samples: dict) -> None:
    selections = (
        ("flow", 10, "Flow Matching, 10 steps"),
        ("ddim", 10, "DDIM, 10 steps"),
        ("ddpm", 10, "DDPM, 10 steps"),
        ("ddpm", 1000, "DDPM, 1000 steps"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    for axis, (method, steps, title) in zip(axes, selections):
        points = samples[method][str(steps)]
        axis.scatter(points[:, 0], points[:, 1], s=5, alpha=0.35, rasterized=True)
        axis.scatter(
            TARGET_CENTRES[:, 0], TARGET_CENTRES[:, 1], marker="x", s=80,
            linewidths=2, color="black", label="target centres",
        )
        axis.set_xlim(-8, 8)
        axis.set_ylim(-8, 8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    figure.suptitle("Generated distributions; fixed display range [-8, 8]")
    figure.savefig(DISTRIBUTION_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    diffusion_parameters = load_diffusion_checkpoint(DIFFUSION_CHECKPOINT)
    flow_parameters = load_flow_checkpoint(FLOW_CHECKPOINT)
    root_key = jax.random.key(SEED)
    initial_key, reference_key, noise_key = jax.random.split(root_key, 3)
    initial_noise = jax.random.normal(
        initial_key, (SAMPLE_COUNT, 2), dtype=jnp.float32
    )
    reference, _ = sample_target(reference_key, SAMPLE_COUNT)
    reference = np.asarray(reference)
    reference_self = mean_pairwise_distance(reference, reference)
    results = {"flow": [], "ddim": [], "ddpm": []}
    sample_store = {"flow": {}, "ddim": {}, "ddpm": {}}
    noise_keys = jax.random.split(noise_key, len(DDPM_DDIM_STEPS))

    for steps in STEP_COUNTS:
        flow_times = tuple(np.linspace(1.0, 0.0, steps + 1, dtype=np.float32))
        flow_states = np.asarray(flow_trajectory(flow_parameters, initial_noise, flow_times))
        flow_samples = flow_states[-1]
        flow_stats = distribution_stats(flow_samples)
        flow_energy = energy_distance(flow_samples, reference, reference_self)
        results["flow"].append({
            "steps": steps,
            **flow_stats,
            "energy_distance": flow_energy,
            "time_grid": list(map(float, flow_times)),
            "finite": bool(np.all(np.isfinite(flow_samples))),
        })
        sample_store["flow"][str(steps)] = flow_samples

    for index, steps in enumerate(DDPM_DDIM_STEPS):
        levels = integer_levels(steps)
        ddim_samples = np.asarray(ddim_sample(
            diffusion_parameters, jax.random.key(0), SAMPLE_COUNT, levels, initial_noise=initial_noise
        ))
        ddim_stats = distribution_stats(ddim_samples)
        results["ddim"].append({
            "steps": steps,
            **ddim_stats,
            "energy_distance": energy_distance(ddim_samples, reference, reference_self),
            "levels": list(levels),
            "finite": bool(np.all(np.isfinite(ddim_samples))),
        })
        sample_store["ddim"][str(steps)] = ddim_samples

        noises = reverse_noise(noise_keys[index], steps)
        ddpm_samples = np.asarray(
            coarse_ddpm_sample(diffusion_parameters, initial_noise, noises, levels)
        )
        ddpm_stats = distribution_stats(ddpm_samples)
        results["ddpm"].append({
            "steps": steps,
            **ddpm_stats,
            "energy_distance": energy_distance(ddpm_samples, reference, reference_self),
            "levels": list(levels),
            "finite": bool(np.all(np.isfinite(ddpm_samples))),
        })
        sample_store["ddpm"][str(steps)] = ddpm_samples

    adjacent_noise = jax.random.normal(
        jax.random.key(9320), (1000, SAMPLE_COUNT, 2), dtype=jnp.float32
    )
    consistency = adjacent_consistency_check(diffusion_parameters, initial_noise, adjacent_noise)
    plot_comparison(results)
    plot_distributions(sample_store)

    output = {
        "seed": SEED,
        "sample_count": SAMPLE_COUNT,
        "step_counts": list(STEP_COUNTS),
        "ddpm_ddim_step_counts": list(DDPM_DDIM_STEPS),
        "common_initial_noise": True,
        "reference": {
            "distribution": "four-component Gaussian mixture",
            "sample_count": SAMPLE_COUNT,
            "mean": reference.mean(axis=0).tolist(),
            "variance": reference.var(axis=0, ddof=1).tolist(),
        },
        "models": {
            "diffusion_checkpoint": str(DIFFUSION_CHECKPOINT.relative_to(ROOT.parent)),
            "flow_checkpoint": str(FLOW_CHECKPOINT.relative_to(ROOT.parent)),
        },
        "ddpm_adjacent_consistency": consistency,
        "results": results,
        "plots": {
            "metrics": str(FIGURE_PATH.relative_to(ROOT.parent)),
            "distributions": str(DISTRIBUTION_PATH.relative_to(ROOT.parent)),
        },
        "runtime_seconds": time.perf_counter() - started,
        "device": str(jax.devices()[0]),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {FIGURE_PATH}")
    print(f"wrote {DISTRIBUTION_PATH}")


if __name__ == "__main__":
    main()
