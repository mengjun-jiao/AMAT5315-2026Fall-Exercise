"""Verify target sampling and the Week 7 forward diffusion definitions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from diffusion import (
    BETA_END,
    BETA_START,
    DIFFUSION_STEPS,
    TARGET_CENTERS,
    TARGET_COMPONENT_STD,
    diffusion_schedule,
    direct_forward_diffusion,
    forward_diffusion_step,
    sample_target,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "artifacts" / "part1_forward_diffusion.json"
SEED = 7101
SAMPLE_COUNT = 20_000
TOLERANCE = 0.10


def check_close(name: str, value: float, expected: float, tolerance: float) -> None:
    if not np.isfinite(value) or abs(value - expected) > tolerance:
        raise AssertionError(f"{name}={value} is not within {tolerance} of {expected}")


def main() -> None:
    started = time.perf_counter()
    key = jax.random.key(SEED)
    samples, components = sample_target(key, SAMPLE_COUNT)
    samples_np = np.asarray(samples)
    components_np = np.asarray(components)

    if samples_np.shape != (SAMPLE_COUNT, 2):
        raise AssertionError(f"unexpected target sample shape: {samples_np.shape}")
    frequencies = np.bincount(components_np, minlength=4) / SAMPLE_COUNT
    component_means = np.asarray(
        [samples_np[components_np == index].mean(axis=0) for index in range(4)]
    )
    component_stds = np.asarray(
        [samples_np[components_np == index].std(axis=0, ddof=1) for index in range(4)]
    )
    if np.max(np.abs(frequencies - 0.25)) > 0.03:
        raise AssertionError(f"component frequencies are not sufficiently balanced: {frequencies}")
    if np.max(np.abs(component_means - np.asarray(TARGET_CENTERS))) > TOLERANCE:
        raise AssertionError(f"component means are too far from target centers: {component_means}")
    if np.max(np.abs(component_stds - TARGET_COMPONENT_STD)) > TOLERANCE:
        raise AssertionError(f"component standard deviations are too far from 0.3: {component_stds}")

    betas, alphas, alpha_bars = diffusion_schedule()
    betas_np = np.asarray(betas)
    alpha_bars_np = np.asarray(alpha_bars)
    if betas.shape != (DIFFUSION_STEPS,):
        raise AssertionError(f"unexpected beta shape: {betas.shape}")
    if not np.isclose(betas_np[0], BETA_START) or not np.isclose(betas_np[-1], BETA_END):
        raise AssertionError("beta endpoints are incorrect")
    if not np.all(np.diff(betas_np) > 0.0):
        raise AssertionError("beta schedule is not strictly increasing")
    if alpha_bars.shape != (DIFFUSION_STEPS,):
        raise AssertionError(f"unexpected alpha_bar shape: {alpha_bars.shape}")
    if not np.all(np.isfinite(alpha_bars_np)) or not np.all(alpha_bars_np > 0.0):
        raise AssertionError("alpha_bar contains invalid values")
    if not np.all(np.diff(alpha_bars_np) < 0.0):
        raise AssertionError("alpha_bar schedule is not strictly decreasing")

    x_zero = samples[:8]
    direct_key = jax.random.key(SEED + 1)
    direct_noise = jax.random.normal(direct_key, shape=x_zero.shape, dtype=x_zero.dtype)
    direct_result = direct_forward_diffusion(x_zero, direct_key, DIFFUSION_STEPS, alpha_bars)
    direct_expected = (
        jnp.sqrt(alpha_bars[-1]) * x_zero
        + jnp.sqrt(1.0 - alpha_bars[-1]) * direct_noise
    )
    direct_formula_error = float(jnp.max(jnp.abs(direct_result - direct_expected)))
    if direct_formula_error > 1e-6:
        raise AssertionError(f"direct formula mismatch: {direct_formula_error}")

    step_keys = jax.random.split(jax.random.key(SEED + 2), DIFFUSION_STEPS)
    sequential = x_zero
    for step in range(1, DIFFUSION_STEPS + 1):
        sequential = forward_diffusion_step(sequential, step_keys[step - 1], step, betas, alphas)
    effective_epsilon = (
        sequential - jnp.sqrt(alpha_bars[-1]) * x_zero
    ) / jnp.sqrt(1.0 - alpha_bars[-1])
    sequential_reconstructed = (
        jnp.sqrt(alpha_bars[-1]) * x_zero
        + jnp.sqrt(1.0 - alpha_bars[-1]) * effective_epsilon
    )
    schedule_consistency_error = float(jnp.max(jnp.abs(sequential - sequential_reconstructed)))
    if schedule_consistency_error > 1e-5:
        raise AssertionError(f"step/direct schedule mismatch: {schedule_consistency_error}")

    t1000_key = jax.random.key(SEED + 3)
    t1000 = direct_forward_diffusion(samples, t1000_key, DIFFUSION_STEPS, alpha_bars)
    t1000_np = np.asarray(t1000)
    t1000_mean = t1000_np.mean(axis=0)
    t1000_variance = t1000_np.var(axis=0, ddof=1)
    theoretical_variance = 1.0 + float(alpha_bars[-1]) * (
        np.var(samples_np, axis=0, ddof=1) - 1.0
    )
    if np.max(np.abs(t1000_mean)) > 0.12:
        raise AssertionError(f"t=1000 mean is too far from zero: {t1000_mean}")
    if np.max(np.abs(t1000_variance - theoretical_variance)) > 0.12:
        raise AssertionError(
            f"t=1000 variance does not match theory: {t1000_variance} vs {theoretical_variance}"
        )

    output = {
        "seed": SEED,
        "target_sampling": {
            "sample_count": SAMPLE_COUNT,
            "sample_shape": list(samples_np.shape),
            "centers": np.asarray(TARGET_CENTERS).tolist(),
            "component_weights": [0.25] * 4,
            "component_standard_deviation": TARGET_COMPONENT_STD,
            "component_frequencies": frequencies.tolist(),
            "component_means": component_means.tolist(),
            "component_standard_deviations": component_stds.tolist(),
        },
        "forward_diffusion": {
            "steps": DIFFUSION_STEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "beta_first": float(betas_np[0]),
            "beta_last": float(betas_np[-1]),
            "alpha_bar_first": float(alpha_bars_np[0]),
            "alpha_bar_last": float(alpha_bars_np[-1]),
            "direct_formula_max_error": direct_formula_error,
            "step_direct_schedule_max_error": schedule_consistency_error,
            "t1000": {
                "mean": t1000_mean.tolist(),
                "variance": t1000_variance.tolist(),
                "theoretical_variance": theoretical_variance.tolist(),
            },
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"target shape: {samples_np.shape}")
    print(f"component frequencies: {frequencies}")
    print(f"beta endpoints: {betas_np[0]:.7f}, {betas_np[-1]:.7f}")
    print(f"alpha_bar endpoints: {alpha_bars_np[0]:.7f}, {alpha_bars_np[-1]:.7f}")
    print(f"t=1000 mean: {t1000_mean}")
    print(f"t=1000 variance: {t1000_variance}")
    print(f"t=1000 theoretical variance: {theoretical_variance}")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
