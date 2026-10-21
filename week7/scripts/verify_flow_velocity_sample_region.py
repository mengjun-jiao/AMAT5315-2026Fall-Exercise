"""Verify the Flow Matching velocity field on interpolation-path samples."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from diffusion import sample_target
from exact_velocity import exact_marginal_velocity
from flow_model import predict_velocity
from verify_flow_velocity import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = ROOT / "artifacts" / "part3_flow_training.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "checkpoints" / "flow_update_025000.npz"
OUTPUT_PATH = ROOT / "artifacts" / "part3_flow_velocity_sample_region.json"
FIGURE_PATH = ROOT / "evidence" / "part3_flow_velocity_sample_region.png"

SEED = 9107
SAMPLE_COUNT = 100_000
TEST_TIMES = (0.1, 0.5, 0.9)
GRID_SIZE = 41
BOUND_QUANTILES = (0.01, 0.99)
REFERENCE_ERRORS = {"0.1": 0.011, "0.5": 0.017, "0.9": 0.058}


def interpolation_samples(key: jax.Array, s: float) -> np.ndarray:
    data_key, noise_key = jax.random.split(key)
    x_zero, _ = sample_target(data_key, SAMPLE_COUNT)
    noise = jax.random.normal(noise_key, (SAMPLE_COUNT, 2), dtype=jnp.float32)
    return np.asarray((1.0 - s) * x_zero + s * noise)


def field_values(parameters: dict, points: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    points_array = jnp.asarray(points, dtype=jnp.float32)
    time = jnp.asarray(s, dtype=jnp.float32)
    learned = np.asarray(predict_velocity(parameters, points_array, time))
    exact = np.asarray(exact_marginal_velocity(points_array, time))
    return learned, exact


def metrics(learned: np.ndarray, exact: np.ndarray) -> dict:
    difference = learned - exact
    absolute_error = np.linalg.norm(difference, axis=1)
    numerator = float(np.sqrt(np.mean(np.square(difference))))
    denominator = float(np.sqrt(np.mean(np.square(exact))))
    relative = numerator / denominator
    return {
        "rms_numerator": numerator,
        "exact_field_rms_denominator": denominator,
        "relative_rms": relative,
        "mean_absolute_velocity_error": float(np.mean(absolute_error)),
        "maximum_absolute_velocity_error": float(np.max(absolute_error)),
        "criterion_relative_rms_below_0.1": relative < 0.1,
    }


def make_plot(fields: dict[str, dict], grid_size: int) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    for row, s in enumerate(TEST_TIMES):
        entry = fields[str(s)]
        grid = entry["grid_points"]
        limits = entry["grid_limits"]
        axis_values_x = np.linspace(limits[0][0], limits[0][1], grid_size, dtype=np.float32)
        axis_values_y = np.linspace(limits[1][0], limits[1][1], grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(axis_values_x, axis_values_y)
        for column, (name, title) in enumerate(
            (("learned_grid", "Learned"), ("exact_grid", "Exact"))
        ):
            axis = axes[row, column]
            field = np.asarray(entry[name])
            axis.quiver(
                grid_x,
                grid_y,
                field[:, 0].reshape(grid_x.shape),
                field[:, 1].reshape(grid_y.shape),
                angles="xy",
                scale_units="xy",
                scale=12,
            )
            axis.set_xlim(*limits[0])
            axis.set_ylim(*limits[1])
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_title(f"{title} velocity, s={s}")
            axis.grid(alpha=0.2)
    figure.suptitle("Flow Matching velocity fields on sample-supported grids")
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)


def main() -> None:
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    parameters = load_checkpoint(CHECKPOINT_PATH)
    time_keys = jax.random.split(jax.random.key(SEED), len(TEST_TIMES))
    fields: dict[str, dict] = {}
    results: dict[str, dict] = {}

    for s, time_key in zip(TEST_TIMES, time_keys):
        samples = interpolation_samples(time_key, s)
        lower = np.quantile(samples, BOUND_QUANTILES[0], axis=0).astype(np.float32)
        upper = np.quantile(samples, BOUND_QUANTILES[1], axis=0).astype(np.float32)
        x_values = np.linspace(lower[0], upper[0], GRID_SIZE, dtype=np.float32)
        y_values = np.linspace(lower[1], upper[1], GRID_SIZE, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        grid_points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)

        learned_samples, exact_samples = field_values(parameters, samples, s)
        learned_grid, exact_grid = field_values(parameters, grid_points, s)
        sample_metrics = metrics(learned_samples, exact_samples)
        results[str(s)] = {
            "sample_evaluation": {
                "point_count": SAMPLE_COUNT,
                **sample_metrics,
            },
            "visualization_grid": {
                "point_count": int(grid_points.shape[0]),
                "resolution": [GRID_SIZE, GRID_SIZE],
                "x_bounds": [float(lower[0]), float(upper[0])],
                "y_bounds": [float(lower[1]), float(upper[1])],
            },
            "course_reference_relative_rms": REFERENCE_ERRORS[str(s)],
            "absolute_difference_from_course_reference": abs(
                sample_metrics["relative_rms"] - REFERENCE_ERRORS[str(s)]
            ),
        }
        fields[str(s)] = {
            "grid_limits": ([float(lower[0]), float(upper[0])], [float(lower[1]), float(upper[1])]),
            "learned_grid": learned_grid,
            "exact_grid": exact_grid,
            "grid_points": grid_points,
        }

    make_plot(fields, GRID_SIZE)
    output = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT.parent)),
        "training_config": training["config"],
        "training_updates": training["updates"],
        "seed": SEED,
        "sample_count": SAMPLE_COUNT,
        "time_points": list(TEST_TIMES),
        "evaluation_point_construction": {
            "formula": "x_s = (1-s) * x0 + s * z",
            "x0_distribution": "existing four-component Gaussian mixture",
            "z_distribution": "standard Gaussian in R^2",
            "independent_subkeys_per_time": True,
            "numerical_evaluation_points": "100000 interpolation-path samples",
        },
        "visualization_grid": {
            "resolution": [GRID_SIZE, GRID_SIZE],
            "bounds_method": "coordinate-wise 1% and 99% quantiles of the same-time interpolation samples",
            "quantiles": list(BOUND_QUANTILES),
            "separate_from_numerical_evaluation_points": True,
        },
        "results": results,
        "official_rectangular_grid_unchanged": True,
        "figure": str(FIGURE_PATH.relative_to(ROOT.parent)),
        "device": str(jax.devices()[0]),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
