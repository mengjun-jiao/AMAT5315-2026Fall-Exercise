"""Run finite-output smoke tests for the initial and trained DDPM models."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from ddpm import ddpm_sample
from noise_predictor import HIDDEN_WIDTHS, initialize_parameters
from train_diffusion import load_config


ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = ROOT / "artifacts" / "part1_diffusion_training.json"
OUTPUT_PATH = ROOT / "artifacts" / "part1_ddpm_smoke_test.json"
PREVIEW_PATH = ROOT / "evidence" / "part1_ddpm_smoke_test.png"
SAMPLE_COUNT = 16
SEED = 2027


def load_checkpoint(path: Path) -> dict:
    data = np.load(path)
    weights = tuple(
        np.asarray(data[f"weight_{index}"], dtype=np.float32)
        for index in range(len(HIDDEN_WIDTHS) + 1)
    )
    biases = tuple(
        np.asarray(data[f"bias_{index}"], dtype=np.float32)
        for index in range(len(HIDDEN_WIDTHS) + 1)
    )
    return {"weights": weights, "biases": biases}


def main() -> None:
    started = time.perf_counter()
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    checkpoint_paths = {
        int(update): ROOT / path for update, path in training["checkpoint_files"].items()
    }
    key = jax.random.key(SEED)
    results = {}
    samples_for_preview = None
    for update in (0, 25000):
        parameters = load_checkpoint(checkpoint_paths[update])
        key, sample_key = jax.random.split(key)
        samples = np.asarray(ddpm_sample(parameters, sample_key, SAMPLE_COUNT))
        finite = bool(np.all(np.isfinite(samples)))
        shape_ok = samples.shape == (SAMPLE_COUNT, 2)
        if not finite or not shape_ok:
            raise AssertionError(f"invalid DDPM output at update {update}: {samples.shape}")
        results[str(update)] = {
            "shape": list(samples.shape),
            "finite": finite,
            "mean": samples.mean(axis=0).tolist(),
            "variance": samples.var(axis=0, ddof=1).tolist(),
        }
        if update == 25000:
            samples_for_preview = samples

    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.scatter(samples_for_preview[:, 0], samples_for_preview[:, 1], s=35, alpha=0.8)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("DDPM smoke test: update 25000")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    figure.savefig(PREVIEW_PATH, dpi=160)
    plt.close(figure)

    output = {
        "seed": SEED,
        "sample_count": SAMPLE_COUNT,
        "steps": 1000,
        "models": results,
        "preview": str(PREVIEW_PATH.relative_to(ROOT.parent)),
        "runtime_seconds": time.perf_counter() - started,
        "device": str(jax.devices()[0]),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {PREVIEW_PATH}")


if __name__ == "__main__":
    main()
