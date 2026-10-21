"""Compare the trained Flow Matching field with the exact marginal field."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from exact_velocity import exact_marginal_velocity
from flow_model import INPUT_WIDTH, predict_velocity


ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = ROOT / "artifacts" / "part3_flow_training.json"
CHECKPOINT_PATH = ROOT / "artifacts" / "checkpoints" / "flow_update_025000.npz"
OUTPUT_PATH = ROOT / "artifacts" / "part3_flow_velocity_verification.json"
FIGURE_PATH = ROOT / "evidence" / "part3_flow_velocity_fields.png"
TEST_TIMES = (0.1, 0.5, 0.9)
GRID_LIMITS = (-4.5, 4.5)
GRID_SIZE = 41


def load_checkpoint(path: Path) -> dict:
    data = np.load(path)
    layer_count = len([name for name in data.files if name.startswith("weight_")])
    return {
        "weights": tuple(
            jnp.asarray(data[f"weight_{index}"], dtype=jnp.float32)
            for index in range(layer_count)
        ),
        "biases": tuple(
            jnp.asarray(data[f"bias_{index}"], dtype=jnp.float32)
            for index in range(layer_count)
        ),
    }


def main() -> None:
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    parameters = load_checkpoint(CHECKPOINT_PATH)
    axis_values = np.linspace(*GRID_LIMITS, GRID_SIZE, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis_values, axis_values)
    grid = jnp.asarray(np.stack((grid_x.ravel(), grid_y.ravel()), axis=1))

    dimension_check = {
        "input_width": int(parameters["weights"][0].shape[0]),
        "output_width": int(parameters["weights"][-1].shape[1]),
        "expected_input_width": INPUT_WIDTH + 32,
        "expected_output_width": 2,
        "hidden_widths": [int(weight.shape[1]) for weight in parameters["weights"][:-1]],
    }
    errors = {}
    finite_checks = {}
    fields = {}
    for value in (0.0, *TEST_TIMES, 1.0):
        time_array = jnp.asarray(value, dtype=jnp.float32)
        learned = predict_velocity(parameters, grid, time_array)
        exact = exact_marginal_velocity(grid, time_array)
        finite_checks[str(value)] = {
            "learned": bool(np.all(np.isfinite(np.asarray(learned)))),
            "exact": bool(np.all(np.isfinite(np.asarray(exact)))),
        }
        if value in TEST_TIMES:
            learned_np = np.asarray(learned)
            exact_np = np.asarray(exact)
            errors[str(value)] = float(
                np.sqrt(np.mean(np.square(learned_np - exact_np)))
                / np.sqrt(np.mean(np.square(exact_np)))
            )
            fields[str(value)] = {
                "learned": learned_np,
                "exact": exact_np,
            }

    endpoint_exact = np.asarray(exact_marginal_velocity(grid, jnp.asarray(1.0)))
    endpoint_error = float(np.max(np.abs(endpoint_exact - np.asarray(grid))))
    history_length = len(training["loss_history"])
    acceptance = {
        "relative_rms_below_0.1_at_all_test_times": all(error < 0.1 for error in errors.values()),
        "finite_at_all_requested_times": all(
            item["learned"] and item["exact"] for item in finite_checks.values()
        ),
        "exact_endpoint_matches_x": endpoint_error < 1e-6,
        "training_history_has_25000_updates": history_length == 25000,
    }

    figure, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    for row, value in enumerate(TEST_TIMES):
        learned = fields[str(value)]["learned"]
        exact = fields[str(value)]["exact"]
        for column, (field, title) in enumerate(((learned, "Learned"), (exact, "Exact"))):
            axis = axes[row, column]
            field_x = field[:, 0].reshape(grid_x.shape)
            field_y = field[:, 1].reshape(grid_y.shape)
            axis.quiver(grid_x, grid_y, field_x, field_y, angles="xy", scale_units="xy", scale=12)
            axis.set_xlim(*GRID_LIMITS)
            axis.set_ylim(*GRID_LIMITS)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_title(f"{title} velocity, s={value}")
            axis.grid(alpha=0.2)
    figure.suptitle("Flow Matching velocity fields")
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)

    output = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT.parent)),
        "grid": {"size": GRID_SIZE, "limits": list(GRID_LIMITS), "points": int(grid.shape[0])},
        "test_times": list(TEST_TIMES),
        "relative_rms_errors": errors,
        "finite_checks": finite_checks,
        "exact_endpoint_max_absolute_error": endpoint_error,
        "dimension_check": dimension_check,
        "training_history_length": history_length,
        "acceptance": acceptance,
        "device": str(jax.devices()[0]),
        "figure": str(FIGURE_PATH.relative_to(ROOT.parent)),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
