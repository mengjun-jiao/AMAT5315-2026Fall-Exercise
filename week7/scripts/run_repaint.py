"""Run RePaint on the four prepared course test images."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from cifar_unet import CifarUNet2D
from load_cifar_checkpoint import load_safetensors, read_config
from repaint import repaint_sample


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "inputs" / "images.npz"
MODEL_DIR = ROOT / "inputs" / "cifar-model"
CONFIG_PATH = MODEL_DIR / "config.json"
WEIGHTS_PATH = MODEL_DIR / "diffusion_pytorch_model.safetensors"
JSON_PATH = ROOT / "artifacts" / "part5_repaint_results.json"
RAW_PATH = ROOT / "artifacts" / "part5_repaint_raw.npz"
FIGURE_PATH = ROOT / "evidence" / "part5_repaint_completions.png"

SEEDS = (7201, 7202, 7203, 7204)
LEVELS = tuple(range(1000, -1, -20))
ROW_START, ROW_END = 6, 26
COL_START, COL_END = 6, 26


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display(image: np.ndarray) -> np.ndarray:
    return np.clip((image + 1.0) / 2.0, 0.0, 1.0).transpose(1, 2, 0)


def main() -> None:
    started = time.perf_counter()
    with np.load(INPUT_PATH) as data:
        if "test" not in data.files:
            raise ValueError(f"input must contain test key, found {data.files}")
        originals = np.asarray(data["test"], dtype=np.float32).copy()
    if originals.shape != (4, 3, 32, 32):
        raise ValueError(f"expected test shape (4,3,32,32), got {originals.shape}")
    if not np.isfinite(originals).all() or originals.min() < -1.0 or originals.max() > 1.0:
        raise ValueError("test images must be finite and lie in [-1, 1]")

    observed_mask = np.ones((3, 32, 32), dtype=bool)
    observed_mask[:, ROW_START:ROW_END, COL_START:COL_END] = False
    missing_pixels = int((~observed_mask).sum())
    if missing_pixels != 3 * 20 * 20:
        raise AssertionError(f"unexpected missing pixel count: {missing_pixels}")
    observed = originals * observed_mask[None, ...]

    config = read_config(CONFIG_PATH)
    tensors, _, tensor_names = load_safetensors(WEIGHTS_PATH)
    model = CifarUNet2D(tensors, config)
    weights_before = sha256(WEIGHTS_PATH)
    completions = []
    per_image = []
    for index, seed in enumerate(SEEDS):
        completion = np.asarray(
            repaint_sample(model, observed[index], observed_mask, seed, LEVELS),
            dtype=np.float32,
        )
        completions.append(completion)
        hidden = ~observed_mask
        hidden_mse = float(np.mean(np.square(completion[hidden] - originals[index][hidden])))
        observed_error = float(np.max(np.abs(completion[observed_mask] - originals[index][observed_mask])))
        per_image.append(
            {
                "index": index,
                "seed": seed,
                "shape": list(completion.shape),
                "finite": bool(np.isfinite(completion).all()),
                "hidden_pixel_mse": hidden_mse,
                "observed_pixel_max_abs_change": observed_error,
                "observed_pixels_preserved": observed_error < 1e-5,
            }
        )
    completions = np.stack(completions, axis=0)
    weights_after = sha256(WEIGHTS_PATH)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW_PATH, completions=completions, observed=observed, mask=observed_mask)

    variance = np.var(completions, axis=0, ddof=0).mean(axis=0)
    figure, axes = plt.subplots(4, 4, figsize=(12, 12))
    column_titles = ("Original", "Observed", "RePaint completion", "Completion variance")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)
    for index in range(4):
        axes[index, 0].imshow(display(originals[index]))
        axes[index, 1].imshow(display(observed[index]))
        axes[index, 2].imshow(display(completions[index]))
        axes[index, 3].imshow(variance, cmap="magma")
        axes[index, 2].set_xlabel(f"hidden MSE={per_image[index]['hidden_pixel_mse']:.5f}")
        for column in range(4):
            axes[index, column].axis("off")
    figure.suptitle(
        "RePaint central-square completion; variance is per-pixel variance across the four completions\n"
        "display only: raw [-1, 1] values mapped to [0, 1]"
    )
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)

    result = {
        "input": {
            "path": str(INPUT_PATH.relative_to(ROOT)),
            "key": "test",
            "shape": list(originals.shape),
            "dtype": str(originals.dtype),
            "range": [float(originals.min()), float(originals.max())],
            "finite": bool(np.isfinite(originals).all()),
        },
        "model": {
            "repository": "google/ddpm-cifar10-32",
            "intended_revision": "267b167dc01f0e4e61923ea244e8b988f84deb80",
            "config_sha256": sha256(CONFIG_PATH),
            "weights_sha256": weights_before,
            "weights_sha256_after_sampling": weights_after,
            "weights_unchanged": weights_before == weights_after,
            "checkpoint_tensor_count": len(tensor_names),
        },
        "sampling": {
            "device": str(jax.devices()[0]),
            "dtype": "float32",
            "levels": list(LEVELS),
            "reverse_jumps": len(LEVELS) - 1,
            "attempts_per_jump": "two except final jump to zero, which has one",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule_steps": 1000,
            "timestep_convention": "mathematical t, with model input t mapped internally to t-1",
            "clean_estimate_clipped_only": True,
            "noisy_states_clipped": False,
            "runtime_seconds": time.perf_counter() - started,
        },
        "mask": {
            "observed_mask_true_outside_square": True,
            "hidden_rows": [ROW_START, ROW_END],
            "hidden_columns": [COL_START, COL_END],
            "hidden_square_shape": [20, 20],
            "hidden_pixel_count_all_channels": missing_pixels,
        },
        "images": per_image,
        "outputs": {
            "raw_path": str(RAW_PATH.relative_to(ROOT)),
            "figure_path": str(FIGURE_PATH.relative_to(ROOT)),
            "completion_shape": list(completions.shape),
            "all_finite": bool(np.isfinite(completions).all()),
            "max_observed_pixel_error": max(item["observed_pixel_max_abs_change"] for item in per_image),
            "all_observed_pixels_preserved": all(item["observed_pixels_preserved"] for item in per_image),
            "variance_definition": "population variance at each channel and pixel across the four completion arrays, averaged over channels for display",
        },
    }
    JSON_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mse": [item["hidden_pixel_mse"] for item in per_image], "observed_errors": [item["observed_pixel_max_abs_change"] for item in per_image], "runtime_seconds": result["sampling"]["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
