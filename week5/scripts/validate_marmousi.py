#!/usr/bin/env python3
"""Validate the official Marmousi Born and Treeverse adjoint outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_checkpoints import audit_actions


def main() -> None:
    experiment = json.loads((ROOT / "inputs/marmousi.json").read_text())
    nx = experiment["nx"]
    nz = experiment["nz"]
    steps = experiment["steps"]
    shots = len(experiment["shots"])
    receivers = len(experiment["receivers"])
    assert (nx, nz, steps, shots, receivers) == (805, 269, 1200, 9, 91)

    born_dir = ROOT / "artifacts/marmousi-born"
    image_dir = ROOT / "artifacts/marmousi-image"
    born_run = json.loads((born_dir / "run.json").read_text())
    born_result = json.loads((born_dir / "result.json").read_text())
    image_run = json.loads((image_dir / "run.json").read_text())
    image_result = json.loads((image_dir / "result.json").read_text())
    assert born_run["experiment_file"] == "inputs/marmousi.json"
    assert image_run["experiment_file"] == "inputs/marmousi.json"
    for result, mode in ((born_result, "born"), (image_result, "adjoint")):
        assert result["mode"] == mode
        assert result["nx"] == nx and result["nz"] == nz
        assert result["steps"] == steps
        assert len(result["shots"]) == shots
        assert len(result["receivers"]) == receivers

    born = np.load(born_dir / "born_data.npy")
    image = np.load(image_dir / "image.npy")
    assert born.shape == (shots, steps, receivers)
    assert born.dtype == np.float64
    assert np.isfinite(born).all() and np.any(born != 0)
    assert image.shape == (nz, nx)
    assert image.dtype == np.float64
    assert np.isfinite(image).all() and np.any(image != 0)

    born_norm = float(np.linalg.norm(born))
    shot_norms = np.linalg.norm(born, axis=(1, 2))
    born_max_location = tuple(int(v) for v in np.unravel_index(np.argmax(np.abs(born)), born.shape))
    image_norm = float(np.linalg.norm(image))
    image_reference = 6.7037741e-4
    image_relative_error = abs(image_norm - image_reference) / image_reference
    assert image_relative_error < 1e-4

    stats = image_result["statistics"]
    one_state_bytes = 2 * nx * nz * 8
    assert one_state_bytes == 3464720
    assert stats["storage"] == "treeverse"
    assert stats["checkpoints"] == 5
    assert stats["scheduler_forward_calls"] == 70956
    assert stats["reverse_calls"] == 10800
    assert stats["peak_saved_states"] == 6
    assert stats["peak_saved_bytes"] == 6 * one_state_bytes
    assert len(stats["per_shot"]) == shots

    print("shot actions forward reverse peak grad_order_errors invalid_restores budget_overruns")
    print("---- ------- ------- ------- ---- ------------------ ----------------- --------------")
    for shot in range(shots):
        audit = audit_actions(image_dir / f"actions-{shot}.json", steps, 5)
        per_shot = stats["per_shot"][shot]
        assert audit["grad_order_errors"] == 0
        assert audit["invalid_restores"] == 0
        assert audit["budget_overruns"] == 0
        assert audit["extended_errors"] == 0, audit
        assert audit["forward_calls"] == 7884
        assert audit["reverse_calls"] == 1200
        assert audit["peak_saved_states"] == 6
        assert per_shot["scheduler_forward_calls"] == 7884
        assert per_shot["reverse_calls"] == 1200
        assert per_shot["peak_saved_states"] == 6
        print(shot, audit["actions"], audit["forward_calls"], audit["reverse_calls"], audit["peak_saved_states"], audit["grad_order_errors"], audit["invalid_restores"], audit["budget_overruns"])

    perturbation = np.asarray(experiment["perturbation"], dtype=np.float64)
    left = float(np.sum(born * born))
    right = float(np.sum(perturbation * image))
    transpose_relative = abs(left - right) / max(abs(left), abs(right))
    assert transpose_relative < 1e-9

    print(f"Born shape={born.shape} dtype={born.dtype} global_l2={born_norm:.16e}")
    print("Born per-shot L2:", " ".join(f"{value:.16e}" for value in shot_norms))
    print(f"Born max_abs={np.max(np.abs(born)):.16e} location={born_max_location}")
    print(f"Image shape={image.shape} dtype={image.dtype} l2={image_norm:.16e}")
    print(f"Image min={np.min(image):.16e} max={np.max(image):.16e} max_abs={np.max(np.abs(image)):.16e}")
    print(f"Image reference_relative_error={image_relative_error:.16e}")
    print(f"Treeverse storage=treeverse checkpoints=5 scheduler_forward_calls=70956 reverse_calls=10800 peak_saved_states=6 peak_saved_bytes={6 * one_state_bytes}")
    print(f"Transpose left={left:.16e} right={right:.16e} relative={transpose_relative:.16e}")
    print("Marmousi Born and Treeverse adjoint validation: passed")


if __name__ == "__main__":
    main()
