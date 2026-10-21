"""Generate four CIFAR-10 images with the local pretrained DDPM U-Net."""

from __future__ import annotations

import hashlib
import json
import tarfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from cifar_unet import CifarUNet2D
from diffusion import diffusion_schedule
from load_cifar_checkpoint import load_safetensors, read_config


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "inputs" / "cifar-model"
CONFIG_PATH = MODEL_DIR / "config.json"
WEIGHTS_PATH = MODEL_DIR / "diffusion_pytorch_model.safetensors"
RAW_PATH = ROOT / "artifacts" / "part4_cifar_generated_raw.npz"
JSON_PATH = ROOT / "artifacts" / "part4_cifar_generation.json"
EVIDENCE_DIR = ROOT / "evidence"
REAL_DATA_PATH = ROOT.parent / "week6" / "inputs" / "cifar10" / "cifar-10-binary.tar.gz"

SEED = 7404
IMAGE_COUNT = 4
IMAGE_SHAPE = (3, 32, 32)
STEPS = 1000
SNAPSHOT_TIMES = (1000, 500, 200, 50, 0)
MODEL_REPOSITORY = "google/ddpm-cifar10-32"
MODEL_REVISION = "267b167dc01f0e4e61923ea244e8b988f84deb80"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_stats(array: np.ndarray) -> dict:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "finite": bool(np.isfinite(array).all()),
    }


def display_image(image: np.ndarray) -> np.ndarray:
    # This copy is only for rendering; sampler arrays remain in raw float32 form.
    return np.clip((image + 1.0) / 2.0, 0.0, 1.0).transpose(1, 2, 0)


def sample_images(model: CifarUNet2D) -> tuple[dict[int, np.ndarray], int]:
    betas, alphas, alpha_bars = diffusion_schedule(STEPS)
    initial_key, reverse_key = jax.random.split(jax.random.key(SEED))
    state = jax.random.normal(initial_key, (IMAGE_COUNT, *IMAGE_SHAPE), dtype=jnp.float32)
    reverse_noise = jax.random.normal(
        reverse_key, (STEPS, IMAGE_COUNT, *IMAGE_SHAPE), dtype=jnp.float32
    )
    @jax.jit
    def reverse_step(current, t, noise):
        index = jnp.asarray(t, dtype=jnp.int32) - 1
        beta_t = betas[index]
        alpha_t = alphas[index]
        alpha_bar_t = alpha_bars[index]
        eps_hat = model(current, jnp.full((IMAGE_COUNT,), t, dtype=jnp.float32))
        mean = (current - beta_t * eps_hat / jnp.sqrt(1.0 - alpha_bar_t)) / jnp.sqrt(alpha_t)
        noise_scale = jnp.where(t == 1.0, 0.0, jnp.sqrt(beta_t))
        return mean + noise_scale * noise
    snapshots = {1000: np.asarray(state)}
    for reverse_index, t in enumerate(range(STEPS, 0, -1)):
        noise = jnp.zeros_like(state) if t == 1 else reverse_noise[reverse_index]
        state = reverse_step(state, jnp.asarray(t, dtype=jnp.float32), noise)
        if t - 1 in (500, 200, 50, 0):
            snapshots[t - 1] = np.asarray(state)
    return snapshots, STEPS


def plot_stages(snapshots: dict[int, np.ndarray]) -> None:
    figure, axes = plt.subplots(IMAGE_COUNT, len(SNAPSHOT_TIMES), figsize=(13, 10))
    for row in range(IMAGE_COUNT):
        for column, t in enumerate(SNAPSHOT_TIMES):
            axes[row, column].imshow(display_image(snapshots[t][row]))
            axes[row, column].set_title(f"image {row + 1}, t={t}")
            axes[row, column].axis("off")
    figure.suptitle("CIFAR-10 DDPM sampling stages")
    figure.tight_layout()
    figure.savefig(EVIDENCE_DIR / "part4_cifar_generation_stages.png", dpi=160)
    plt.close(figure)


def plot_final(final: np.ndarray) -> None:
    figure, axes = plt.subplots(1, IMAGE_COUNT, figsize=(10, 3))
    for index, axis in enumerate(np.asarray(axes).reshape(-1)):
        axis.imshow(display_image(final[index]))
        axis.set_title(f"image {index + 1}")
        axis.axis("off")
    figure.suptitle("Four generated CIFAR-10 images")
    figure.tight_layout()
    figure.savefig(EVIDENCE_DIR / "part4_cifar_generated_images.png", dpi=200)
    plt.close(figure)


def load_real_images() -> np.ndarray | None:
    if not REAL_DATA_PATH.exists():
        return None
    with tarfile.open(REAL_DATA_PATH, "r:gz") as archive:
        member = archive.getmember("cifar-10-batches-bin/test_batch.bin")
        stream = archive.extractfile(member)
        if stream is None:
            return None
        records = np.frombuffer(stream.read(IMAGE_COUNT * 3073), dtype=np.uint8)
    records = records.reshape(IMAGE_COUNT, 3073)
    pixels = records[:, 1:].reshape(IMAGE_COUNT, 3, 1024).reshape(IMAGE_COUNT, 3, 32, 32)
    return pixels.astype(np.float32) / 255.0


def plot_real_vs_generated(real: np.ndarray, final: np.ndarray) -> None:
    figure, axes = plt.subplots(2, IMAGE_COUNT, figsize=(10, 5))
    for index in range(IMAGE_COUNT):
        axes[0, index].imshow(real[index].transpose(1, 2, 0))
        axes[0, index].set_title(f"real {index + 1}")
        axes[1, index].imshow(display_image(final[index]))
        axes[1, index].set_title(f"generated {index + 1}")
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.suptitle("Real CIFAR-10 test images and generated images")
    figure.tight_layout()
    figure.savefig(EVIDENCE_DIR / "part4_cifar_real_vs_generated.png", dpi=200)
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    config = read_config(CONFIG_PATH)
    tensors, _, names = load_safetensors(WEIGHTS_PATH)
    model = CifarUNet2D(tensors, config)
    checkpoint_hash_before = sha256(WEIGHTS_PATH)
    snapshots, steps_executed = sample_images(model)
    final = snapshots[0]
    checkpoint_hash_after = sha256(WEIGHTS_PATH)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RAW_PATH,
        final=final,
        **{f"snapshot_t{t}": snapshots[t] for t in SNAPSHOT_TIMES},
    )
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    plot_stages(snapshots)
    plot_final(final)
    real = load_real_images()
    if real is not None:
        plot_real_vs_generated(real, final)
        real_status = {
            "available": True,
            "source": str(REAL_DATA_PATH.relative_to(ROOT.parent)),
            "figure": "evidence/part4_cifar_real_vs_generated.png",
        }
    else:
        real_status = {"available": False, "reason": "No local CIFAR-10 binary archive found."}

    snapshot_stats = {
        str(t): {f"image_{i + 1}": array_stats(snapshots[t][i]) for i in range(IMAGE_COUNT)}
        for t in SNAPSHOT_TIMES
    }
    output = {
        "model": {
            "repository": MODEL_REPOSITORY,
            "intended_revision": MODEL_REVISION,
            "config_path": str(CONFIG_PATH.relative_to(ROOT)),
            "weights_path": str(WEIGHTS_PATH.relative_to(ROOT)),
            "config_sha256": sha256(CONFIG_PATH),
            "weights_sha256": checkpoint_hash_before,
            "weights_sha256_after_sampling": checkpoint_hash_after,
            "weights_unchanged": checkpoint_hash_before == checkpoint_hash_after,
            "checkpoint_tensor_count": len(names),
        },
        "sampling": {
            "seed": SEED,
            "image_count": IMAGE_COUNT,
            "image_shape": list(IMAGE_SHAPE),
            "dtype": "float32",
            "device": str(jax.devices()[0]),
            "reverse_steps_requested": STEPS,
            "reverse_steps_executed": steps_executed,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule_steps": STEPS,
            "timestep_convention": "mathematical t=1000,...,1; model receives t and maps internally to t-1",
            "snapshot_timesteps": list(SNAPSHOT_TIMES),
            "states_not_clipped": True,
            "final_reverse_step_added_noise": False,
            "raw_output_path": str(RAW_PATH.relative_to(ROOT)),
            "runtime_seconds": time.perf_counter() - started,
        },
        "snapshots": snapshot_stats,
        "final_finite": bool(np.isfinite(final).all()),
        "real_comparison": real_status,
    }
    JSON_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"steps_executed": steps_executed, "final_shape": list(final.shape), "runtime_seconds": output["sampling"]["runtime_seconds"], "real_comparison": real_status}, indent=2))


if __name__ == "__main__":
    main()
