"""Compare deterministic DDIM and stochastic DDPM sampling on the 2-D target."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ddpm import ddpm_trajectory
from ddim import ddim_trajectory
from diffusion import TARGET_CENTERS, diffusion_schedule, sample_target
from noise_predictor import HIDDEN_WIDTHS
from smoke_test_ddpm import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "update_025000.npz"
JSON_PATH = ROOT / "artifacts" / "part2_ddim_evaluation.json"
TRAJECTORY_PATH = ROOT / "evidence" / "part2_ddim_ddpm_trajectories.png"
SAMPLES_PATH = ROOT / "evidence" / "part2_ddim_ddpm_samples.png"

SEED = 7602
SAMPLE_COUNT = 2000
TRAJECTORY_COUNT = 10
STEPS = 1000
LEVELS = tuple(range(STEPS, -1, -1))
PLOT_LIMITS = (-8.0, 8.0)


def nearest_components(samples: np.ndarray) -> np.ndarray:
    centers = np.asarray(TARGET_CENTERS)
    distances = np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def distribution_stats(samples: np.ndarray) -> dict:
    centers = np.asarray(TARGET_CENTERS)
    components = nearest_components(samples)
    clusters = []
    for component, target in enumerate(centers):
        cluster = samples[components == component]
        if len(cluster) == 0:
            actual_center = [float("nan"), float("nan")]
            spread = [float("nan"), float("nan")]
            centre_error = float("nan")
        else:
            actual_center = cluster.mean(axis=0)
            spread = cluster.std(axis=0, ddof=1)
            centre_error = float(np.linalg.norm(actual_center - target))
        clusters.append(
            {
                "target_center": target.tolist(),
                "count": int(len(cluster)),
                "population": float(len(cluster) / len(samples)),
                "centre": np.asarray(actual_center).tolist(),
                "centre_error": centre_error,
                "spread": np.asarray(spread).tolist(),
            }
        )
    return {
        "clusters": clusters,
        "mean": samples.mean(axis=0).tolist(),
        "variance": samples.var(axis=0, ddof=1).tolist(),
    }


def plot_centres(axis: plt.Axes) -> None:
    centres = np.asarray(TARGET_CENTERS)
    axis.scatter(
        centres[:, 0], centres[:, 1], marker="x", s=90, linewidths=2,
        color="black", label="target centres", zorder=4,
    )


def configure_axis(axis: plt.Axes) -> None:
    axis.set_xlim(*PLOT_LIMITS)
    axis.set_ylim(*PLOT_LIMITS)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(alpha=0.2)


def make_trajectory_plot(
    ddim_states: np.ndarray,
    ddpm_one_states: np.ndarray,
    ddpm_two_states: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    descriptions = (
        (ddim_states, "DDIM"),
        (ddpm_one_states, "DDPM run 1"),
        (ddpm_two_states, "DDPM run 2"),
    )
    for axis, (states, title) in zip(axes, descriptions):
        for index in range(states.shape[1]):
            path = states[:, index, :]
            axis.plot(path[:, 0], path[:, 1], color="tab:blue", alpha=0.45, linewidth=1)
            axis.scatter(path[0, 0], path[0, 1], color="tab:orange", s=20, zorder=3)
            axis.scatter(path[-1, 0], path[-1, 1], color="tab:red", s=28, zorder=3)
        plot_centres(axis)
        configure_axis(axis)
        axis.set_title(title)
    axes[0].scatter([], [], color="tab:orange", s=25, label="initial noise")
    axes[0].scatter([], [], color="tab:red", s=28, label="final sample")
    axes[0].legend(loc="upper right", fontsize=8)
    figure.suptitle("Reverse sampling trajectories for 10 shared initial states")
    figure.savefig(TRAJECTORY_PATH, dpi=160)
    plt.close(figure)


def make_samples_plot(
    target: np.ndarray,
    ddim_samples: np.ndarray,
    ddpm_one: np.ndarray,
    ddpm_two: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    panels = (
        (target, "Exact target"),
        (ddim_samples, "DDIM"),
        (ddpm_one, "DDPM run 1"),
        (ddpm_two, "DDPM run 2"),
    )
    for axis, (samples, title) in zip(axes, panels):
        axis.scatter(samples[:, 0], samples[:, 1], s=7, alpha=0.45, rasterized=True)
        plot_centres(axis)
        configure_axis(axis)
        axis.set_title(title)
    figure.suptitle("2,000 samples; axes fixed to [-8, 8] in both coordinates")
    figure.savefig(SAMPLES_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    parameters = load_checkpoint(CHECKPOINT)
    schedule = diffusion_schedule(steps=STEPS)
    key = jax.random.key(SEED)
    initial_key, ddpm_one_key, ddpm_two_key, target_key = jax.random.split(key, 4)
    initial_noise = jax.random.normal(
        initial_key, (SAMPLE_COUNT, 2), dtype=jnp.float32
    )
    reverse_one = jax.random.normal(
        ddpm_one_key, (STEPS, SAMPLE_COUNT, 2), dtype=jnp.float32
    )
    reverse_two = jax.random.normal(
        ddpm_two_key, (STEPS, SAMPLE_COUNT, 2), dtype=jnp.float32
    )

    ddim_first = np.asarray(ddim_trajectory(parameters, initial_noise, LEVELS)[-1])
    ddim_second = np.asarray(ddim_trajectory(parameters, initial_noise, LEVELS)[-1])
    ddpm_first_states = np.asarray(
        ddpm_trajectory(parameters, initial_noise, reverse_one, STEPS)
    )
    ddpm_second_states = np.asarray(
        ddpm_trajectory(parameters, initial_noise, reverse_two, STEPS)
    )
    ddpm_first = ddpm_first_states[-1]
    ddpm_second = ddpm_second_states[-1]
    target, target_components = sample_target(target_key, SAMPLE_COUNT)
    target = np.asarray(target)

    ddim_difference = float(np.max(np.abs(ddim_first - ddim_second)))
    ddpm_difference = float(np.max(np.abs(ddpm_first - ddpm_second)))
    ddpm_same_cluster = float(
        np.mean(nearest_components(ddpm_first) == nearest_components(ddpm_second))
    )
    stats = {
        "target": distribution_stats(target),
        "ddim": distribution_stats(ddim_first),
        "ddpm_run_1": distribution_stats(ddpm_first),
        "ddpm_run_2": distribution_stats(ddpm_second),
    }
    ddim_population_pass = all(
        abs(cluster["population"] - 0.25) <= 0.015
        for cluster in stats["ddim"]["clusters"]
    )
    generated_spread_pass = all(
        0.29 <= value <= 0.32
        for name in ("ddim", "ddpm_run_1", "ddpm_run_2")
        for cluster in stats[name]["clusters"]
        for value in cluster["spread"]
    )
    finite = all(
        bool(np.all(np.isfinite(values)))
        for values in (ddim_first, ddpm_first, ddpm_second, target)
    )

    make_trajectory_plot(
        np.asarray(ddim_trajectory(parameters, initial_noise[:TRAJECTORY_COUNT], LEVELS)),
        ddpm_first_states[:, :TRAJECTORY_COUNT, :],
        ddpm_second_states[:, :TRAJECTORY_COUNT, :],
    )
    make_samples_plot(target, ddim_first, ddpm_first, ddpm_second)

    output = {
        "seed": SEED,
        "sample_count": SAMPLE_COUNT,
        "trajectory_count": TRAJECTORY_COUNT,
        "checkpoint_update": 25000,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT.parent)),
        "steps": STEPS,
        "ddim_levels": {"count": len(LEVELS), "first": LEVELS[0], "last": LEVELS[-1]},
        "schedule": {
            "beta_start": float(schedule[0][0]),
            "beta_end": float(schedule[0][-1]),
            "alpha_bar_zero": 1.0,
        },
        "shared_initial_noise": True,
        "ddpm_reverse_noise_independent": True,
        "ddim_repeat_max_absolute_difference": ddim_difference,
        "ddpm_repeat_max_absolute_difference": ddpm_difference,
        "ddpm_same_target_cluster_fraction": ddpm_same_cluster,
        "finite_outputs": finite,
        "statistics": stats,
        "acceptance": {
            "ddim_deterministic": ddim_difference <= 1e-6,
            "ddim_population_within_0.015": ddim_population_pass,
            "generated_spreads_within_0.29_0.32": generated_spread_pass,
            "overall_pass": bool(
                finite and ddim_difference <= 1e-6 and ddim_population_pass and generated_spread_pass
            ),
            "ddpm_population_hard_threshold": "not specified by the course",
        },
        "plot_limits": list(PLOT_LIMITS),
        "runtime_seconds": time.perf_counter() - started,
        "device": str(jax.devices()[0]),
        "trajectory_figure": str(TRAJECTORY_PATH.relative_to(ROOT.parent)),
        "samples_figure": str(SAMPLES_PATH.relative_to(ROOT.parent)),
    }
    JSON_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {JSON_PATH}")
    print(f"wrote {TRAJECTORY_PATH}")
    print(f"wrote {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
