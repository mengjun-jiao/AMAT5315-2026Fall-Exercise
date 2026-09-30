"""Validate the official Born/adjoint transpose identity."""

import json
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "inputs" / "reflector.json").open() as handle:
        experiment = json.load(handle)
    born = np.load(root / "artifacts" / "born" / "born_data.npy")
    image = np.load(root / "artifacts" / "adjoint" / "image.npy")
    perturbation = np.asarray(experiment["perturbation"], dtype=np.float64)

    if born.shape != (3, 240, 14) or born.dtype != np.float64:
        raise SystemExit(f"unexpected born_data array: shape={born.shape}, dtype={born.dtype}")
    if image.shape != (41, 41) or image.dtype != np.float64:
        raise SystemExit(f"unexpected image array: shape={image.shape}, dtype={image.dtype}")
    if perturbation.shape != image.shape or perturbation.dtype != np.float64:
        raise SystemExit("perturbation and image shapes/dtypes must match")
    if not np.isfinite(born).all() or not np.isfinite(image).all():
        raise SystemExit("Born data and image must contain only finite values")
    if np.all(born == 0.0) or np.all(image == 0.0):
        raise SystemExit("Born data and image must be nonzero")

    left = float(np.sum(born * born))
    right = float(np.sum(perturbation * image))
    denominator = max(abs(left), abs(right))
    if not np.isfinite(left) or not np.isfinite(right) or denominator == 0.0:
        raise SystemExit("transpose identity has an invalid zero or non-finite scale")
    relative_difference = abs(left - right) / denominator

    print(f"left={left:.15e}")
    print(f"right={right:.15e}")
    print(f"relative_difference={relative_difference:.15e}")
    if relative_difference >= 1.0e-9:
        raise SystemExit("transpose identity validation failed")
    print("transpose identity validation: passed")


if __name__ == "__main__":
    main()
