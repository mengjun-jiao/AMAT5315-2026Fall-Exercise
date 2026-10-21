"""Run forward-pass smoke tests for the local CIFAR UNet checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cifar_unet import CifarUNet2D
from load_cifar_checkpoint import load_safetensors, read_config


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "inputs" / "cifar-model"
OUTPUT_PATH = ROOT / "artifacts" / "part4_cifar_unet_smoke_test.json"


def main() -> None:
    config = read_config(MODEL_DIR / "config.json")
    parameters, header, checkpoint_names = load_safetensors(MODEL_DIR / "diffusion_pytorch_model.safetensors")
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
                "input_shape": list(images.shape),
                "output_shape": list(output.shape),
                "shape_matches": output.shape == images.shape,
                "finite": bool(np.isfinite(output).all()),
                "mean": float(output.mean()),
                "std": float(output.std()),
                "min": float(output.min()),
                "max": float(output.max()),
            })
    unused = sorted(checkpoint_names - model.used_names)
    missing = sorted(model.used_names - checkpoint_names)
    output = {
        "checkpoint": str((MODEL_DIR / "diffusion_pytorch_model.safetensors").relative_to(ROOT.parent)),
        "config": str((MODEL_DIR / "config.json").relative_to(ROOT.parent)),
        "tensor_count": len(checkpoint_names),
        "mapped_tensor_count": len(model.used_names),
        "unused_checkpoint_tensors": unused,
        "missing_checkpoint_tensors": missing,
        "tensor_mapping_complete": not unused and not missing,
        "layout": "NCHW: (batch, channels, height, width)",
        "course_timestep_convention": "mathematical t in 1..1000; model embedding receives t-1",
        "results": results,
        "all_shape_checks_pass": all(item["shape_matches"] for item in results),
        "all_finite_checks_pass": all(item["finite"] for item in results),
        "device": str(jax.devices()[0]),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
