"""Focused validation for the local CIFAR U-Net attention layout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cifar_unet import CifarUNet2D
from load_cifar_checkpoint import load_safetensors, read_config


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "inputs" / "cifar-model"
WEIGHTS_PATH = MODEL_DIR / "diffusion_pytorch_model.safetensors"
OUTPUT_PATH = ROOT / "artifacts" / "part4_cifar_attention_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = read_config(MODEL_DIR / "config.json")
    parameters, _, checkpoint_names = load_safetensors(WEIGHTS_PATH)
    before_hash = sha256(WEIGHTS_PATH)
    model = CifarUNet2D(parameters, config)
    results = []
    key = jax.random.key(9411)
    for batch_size in (1, 4):
        key, input_key = jax.random.split(key)
        images = jax.random.normal(input_key, (batch_size, 3, 32, 32), dtype=jnp.float32)
        for timestep in (1, 500, 1000):
            output = np.asarray(model(images, jnp.asarray(timestep, dtype=jnp.float32)))
            results.append({
                "batch_size": batch_size,
                "timestep": timestep,
                "output_shape": list(output.shape),
                "shape_matches": output.shape == images.shape,
                "finite": bool(np.isfinite(output).all()),
            })
    after_hash = sha256(WEIGHTS_PATH)
    unused = sorted(checkpoint_names - model.used_names)
    missing = sorted(model.used_names - checkpoint_names)
    output = {
        "attention_head_dim_config": config.get("attention_head_dim"),
        "resolved_attention_layouts": model.resolved_attention_layouts,
        "expected_current_layout": "one head of dimension 256 for every attention module",
        "checkpoint_tensor_count": len(checkpoint_names),
        "mapped_tensor_count": len(model.used_names),
        "unused_checkpoint_tensors": unused,
        "missing_checkpoint_tensors": missing,
        "tensor_mapping_complete": not unused and not missing,
        "results": results,
        "all_shape_checks_pass": all(item["shape_matches"] for item in results),
        "all_finite_checks_pass": all(item["finite"] for item in results),
        "weights_sha256_before": before_hash,
        "weights_sha256_after": after_hash,
        "weights_unchanged": before_hash == after_hash,
        "device": str(jax.devices()[0]),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
