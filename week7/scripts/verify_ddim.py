"""Verify deterministic DDIM sampling and the exact mixture noise predictor."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import numpy as np

from ddim import ddim_sample
from exact_noise import exact_conditional_mean_noise
from smoke_test_ddpm import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "artifacts" / "checkpoints" / "update_025000.npz"
OUTPUT_PATH = ROOT / "artifacts" / "part2_ddim_verification.json"
CHECKPOINT_UPDATE = 25000
SEED = 7601
SAMPLE_COUNT = 4
FULL_LEVELS = tuple(range(1000, -1, -1))
SPARSE_LEVELS = (1000, 900, 750, 500, 250, 100, 1, 0)


def finite_and_shape(value: np.ndarray, shape: tuple[int, ...]) -> dict:
    return {
        "shape": list(value.shape),
        "expected_shape": list(shape),
        "shape_ok": value.shape == shape,
        "finite": bool(np.all(np.isfinite(value))),
    }


def main() -> None:
    started = time.perf_counter()
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    parameters = load_checkpoint(CHECKPOINT_PATH)
    initial_noise = np.asarray(
        jax.random.normal(jax.random.key(SEED), (SAMPLE_COUNT, 2), dtype=jax.numpy.float32)
    )

    full_first = np.asarray(
        ddim_sample(parameters, jax.random.key(1), SAMPLE_COUNT, FULL_LEVELS, initial_noise)
    )
    full_second = np.asarray(
        ddim_sample(parameters, jax.random.key(2), SAMPLE_COUNT, FULL_LEVELS, initial_noise)
    )
    sparse_first = np.asarray(
        ddim_sample(parameters, jax.random.key(3), SAMPLE_COUNT, SPARSE_LEVELS, initial_noise)
    )
    sparse_second = np.asarray(
        ddim_sample(parameters, jax.random.key(4), SAMPLE_COUNT, SPARSE_LEVELS, initial_noise)
    )
    single = np.asarray(
        ddim_sample(parameters, jax.random.key(5), 1, SPARSE_LEVELS, initial_noise[0])
    )

    full_repeat_difference = float(np.max(np.abs(full_first - full_second)))
    sparse_repeat_difference = float(np.max(np.abs(sparse_first - sparse_second)))
    full_result = finite_and_shape(full_first, (SAMPLE_COUNT, 2))
    sparse_result = finite_and_shape(sparse_first, (SAMPLE_COUNT, 2))
    single_result = finite_and_shape(single, (2,))
    if not all(result["shape_ok"] and result["finite"] for result in (full_result, sparse_result, single_result)):
        raise AssertionError("DDIM output shape or finite-value check failed")
    if full_repeat_difference > 1e-6 or sparse_repeat_difference > 1e-6:
        raise AssertionError("DDIM repeated runs are not deterministic")

    exact_results = {}
    test_points = np.asarray(initial_noise)
    for level in (1, 500, 1000):
        prediction = np.asarray(exact_conditional_mean_noise(test_points, level))
        result = finite_and_shape(prediction, (SAMPLE_COUNT, 2))
        if not result["shape_ok"] or not result["finite"]:
            raise AssertionError(f"exact noise predictor check failed at t={level}")
        exact_results[str(level)] = result

    output = {
        "checkpoint_update": CHECKPOINT_UPDATE,
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT.parent)),
        "seed": SEED,
        "sample_count": SAMPLE_COUNT,
        "full_levels": list(FULL_LEVELS),
        "sparse_levels": list(SPARSE_LEVELS),
        "alpha_bar_zero": 1.0,
        "ddim": {
            "full_output": full_result,
            "sparse_output": sparse_result,
            "single_output": single_result,
            "full_repeat_max_absolute_difference": full_repeat_difference,
            "sparse_repeat_max_absolute_difference": sparse_repeat_difference,
            "deterministic": full_repeat_difference <= 1e-6 and sparse_repeat_difference <= 1e-6,
            "uses_new_reverse_noise": False,
            "uses_clipping": False,
        },
        "exact_conditional_mean_noise": {
            "levels_checked": [1, 500, 1000],
            "results": exact_results,
            "finite": True,
        },
        "device": str(jax.devices()[0]),
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
