"""Animate the complete forward diffusion trajectory for the Gaussian mixture."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from diffusion import (
    BETA_END,
    BETA_START,
    DIFFUSION_STEPS,
    diffusion_schedule,
    forward_diffusion_step,
    sample_target,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "evidence"
OUTPUT_PATH = ROOT / "artifacts" / "part1_forward_diffusion_animation.json"
GIF_PATH = EVIDENCE_PATH / "part1_forward_diffusion.gif"
STAGES_PATH = EVIDENCE_PATH / "part1_forward_diffusion_stages.png"

SEED = 7201
SAMPLE_COUNT = 2_000
FRAME_STRIDE = 5
STAGE_STEPS = (0, 50, 200, 500, 900, 1000)
FPS = 20


def build_trajectory(
    x_zero: jax.Array,
    keys: jax.Array,
    betas: jax.Array,
    alphas: jax.Array,
    frame_steps: tuple[int, ...],
) -> np.ndarray:
    """Diffuse one fixed batch and retain snapshots at the animation frames."""
    snapshots: dict[int, np.ndarray] = {0: np.asarray(x_zero)}
    state = x_zero
    frame_step_set = set(frame_steps)
    for t in range(1, DIFFUSION_STEPS + 1):
        state = forward_diffusion_step(state, keys[t - 1], t, betas, alphas)
        if t in frame_step_set:
            snapshots[t] = np.asarray(state)
    missing = frame_step_set.difference(snapshots)
    if missing:
        raise RuntimeError(f"trajectory is missing requested frames: {sorted(missing)}")
    return np.stack([snapshots[t] for t in frame_steps])


def fixed_limits(trajectory: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Choose one padded coordinate range for every frame and subplot."""
    extrema = np.max(np.abs(trajectory), axis=(0, 1))
    padding = 0.25
    limits = []
    for value in extrema:
        bound = np.ceil((float(value) + padding) * 10.0) / 10.0
        limits.append((-bound, bound))
    return (limits[0], limits[1])


def stage_statistics(
    trajectory: np.ndarray, frame_steps: tuple[int, ...]
) -> dict[str, dict[str, list[float]]]:
    return {
        str(t): {
            "mean": trajectory[index].mean(axis=0).tolist(),
            "variance": trajectory[index].var(axis=0, ddof=1).tolist(),
        }
        for index, t in enumerate(frame_steps)
    }


def save_stage_figure(
    trajectory: np.ndarray,
    frame_steps: tuple[int, ...],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, state, t in zip(axes.flat, trajectory, frame_steps):
        axis.scatter(state[:, 0], state[:, 1], s=5, alpha=0.55, linewidths=0)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_title(f"t = {t}")
        axis.grid(alpha=0.2)
    figure.suptitle("Forward diffusion stages")
    figure.savefig(STAGES_PATH, dpi=160)
    plt.close(figure)


def save_animation(
    trajectory: np.ndarray,
    frame_steps: tuple[int, ...],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    points = axis.scatter(
        trajectory[0, :, 0], trajectory[0, :, 1], s=7, alpha=0.6, linewidths=0
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(alpha=0.2)
    title = axis.set_title(f"Forward diffusion, t = {frame_steps[0]}")

    def update(frame_index: int):
        state = trajectory[frame_index]
        points.set_offsets(state)
        title.set_text(f"Forward diffusion, t = {frame_steps[frame_index]}")
        return points, title

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_steps),
        interval=1000 / FPS,
        blit=True,
        repeat=True,
    )
    animation.save(GIF_PATH, writer=PillowWriter(fps=FPS))
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    frame_steps = tuple(range(0, DIFFUSION_STEPS + 1, FRAME_STRIDE))
    key = jax.random.key(SEED)
    sample_key, trajectory_key = jax.random.split(key)
    x_zero, components = sample_target(sample_key, SAMPLE_COUNT)
    betas, alphas, alpha_bars = diffusion_schedule()
    step_keys = jax.random.split(trajectory_key, DIFFUSION_STEPS)
    trajectory = build_trajectory(x_zero, step_keys, betas, alphas, frame_steps)
    x_zero_np = np.asarray(x_zero)
    components_np = np.asarray(components)
    x_limits, y_limits = fixed_limits(trajectory)

    stage_indices = [frame_steps.index(t) for t in STAGE_STEPS]
    save_stage_figure(trajectory[stage_indices], STAGE_STEPS, x_limits, y_limits)
    save_animation(trajectory, frame_steps, x_limits, y_limits)

    statistics = stage_statistics(trajectory, frame_steps)
    t1000 = np.asarray(trajectory[-1])
    t1000_mean = t1000.mean(axis=0)
    t1000_variance = t1000.var(axis=0, ddof=1)
    alpha_bar_t1000 = float(alpha_bars[-1])
    empirical_theoretical_variance = (
        1.0 + alpha_bar_t1000 * (x_zero_np.var(axis=0, ddof=1) - 1.0)
    )
    output = {
        "seed": SEED,
        "sample_count": SAMPLE_COUNT,
        "tracked_sample_shape": list(x_zero_np.shape),
        "component_counts": np.bincount(components_np, minlength=4).tolist(),
        "noise_schedule": {
            "steps": DIFFUSION_STEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "alpha_bar_t1000": alpha_bar_t1000,
        },
        "animation": {
            "frame_count": len(frame_steps),
            "frame_stride": FRAME_STRIDE,
            "frame_steps": list(frame_steps),
            "fps": FPS,
            "x_limits": list(x_limits),
            "y_limits": list(y_limits),
        },
        "stage_steps": list(STAGE_STEPS),
        "stage_statistics": statistics,
        "t1000_standard_gaussian_comparison": {
            "sample_mean": t1000_mean.tolist(),
            "sample_variance": t1000_variance.tolist(),
            "theoretical_mean": [0.0, 0.0],
            "theoretical_variance": [1.0, 1.0],
            "mean_error": np.abs(t1000_mean).tolist(),
            "variance_error": np.abs(t1000_variance - 1.0).tolist(),
        },
        "t1000_empirical_initial_distribution_comparison": {
            "theoretical_variance": empirical_theoretical_variance.tolist(),
            "variance_error": np.abs(t1000_variance - empirical_theoretical_variance).tolist(),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"sample shape: {x_zero_np.shape}")
    print(f"animation frames: {len(frame_steps)}")
    print(f"x limits: {x_limits}")
    print(f"y limits: {y_limits}")
    print(f"t=1000 mean: {t1000_mean}")
    print(f"t=1000 variance: {t1000_variance}")
    print(f"t=1000 standard Gaussian variance error: {np.abs(t1000_variance - 1.0)}")
    print(f"wrote {GIF_PATH}")
    print(f"wrote {STAGES_PATH}")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
