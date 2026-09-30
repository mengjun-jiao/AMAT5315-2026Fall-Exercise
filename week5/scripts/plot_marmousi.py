#!/usr/bin/env python3
"""Create the four-panel Marmousi migration evidence figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE_REFERENCE = 6.7037741e-4


def symmetric_limit(values: np.ndarray) -> float:
    limit = float(np.max(np.abs(values)))
    if not np.isfinite(limit) or limit == 0:
        raise ValueError("signed plot data must have a finite nonzero amplitude")
    return limit


def mark_comparison_region(axis: plt.Axes) -> None:
    axis.axvspan(8.0, 14.0, color="white", alpha=0.12, hatch="//", linewidth=0.0)
    axis.axvline(8.0, color="white", linewidth=0.7, alpha=0.8)
    axis.axvline(14.0, color="white", linewidth=0.7, alpha=0.8)


def main() -> None:
    experiment = json.loads((ROOT / "inputs/marmousi.json").read_text())
    nx = int(experiment["nx"])
    nz = int(experiment["nz"])
    dx = float(experiment["dx"])
    dt = float(experiment["dt"])
    length_unit_m = float(experiment["length_unit_m"])
    time_unit_s = float(experiment["time_unit_s"])
    speed_scale = length_unit_m / time_unit_s / 1000.0
    grid_scale_km = dx * length_unit_m / 1000.0

    background = np.asarray(experiment["background"], dtype=np.float64) * speed_scale
    perturbation = np.asarray(experiment["perturbation"], dtype=np.float64) * speed_scale
    born = np.load(ROOT / "artifacts/marmousi-born/born_data.npy", allow_pickle=False)
    image = np.load(ROOT / "artifacts/marmousi-image/image.npy", allow_pickle=False)
    result = json.loads((ROOT / "artifacts/marmousi-image/result.json").read_text())

    assert background.shape == (nz, nx)
    assert perturbation.shape == (nz, nx)
    assert born.shape == (len(experiment["shots"]), experiment["steps"], len(experiment["receivers"]))
    assert image.shape == (nz, nx)
    assert image.dtype == np.float64
    assert np.isfinite(image).all()

    image_l2 = float(np.linalg.norm(image))
    image_relative_error = abs(image_l2 - IMAGE_REFERENCE) / IMAGE_REFERENCE
    assert image_relative_error < 1e-4
    stats = result["statistics"]
    assert stats["peak_saved_states"] == 6
    assert stats["peak_saved_bytes"] == 20788320
    print(f"image_shape={image.shape} image_dtype={image.dtype}")
    print(f"image_l2={image_l2:.16e}")
    print(f"image_min={np.min(image):.16e} image_max={np.max(image):.16e} image_max_abs={np.max(np.abs(image)):.16e}")
    print(f"image_reference_relative_error={image_relative_error:.16e}")
    print(f"peak_saved_states={stats['peak_saved_states']} peak_saved_bytes={stats['peak_saved_bytes']}")

    shots = np.asarray(experiment["shots"], dtype=np.float64)
    target_x = 10.0
    selected = np.flatnonzero(np.isclose(shots[:, 0] * grid_scale_km, target_x, rtol=0.0, atol=1e-12))
    if len(selected) != 1:
        raise ValueError(f"expected one shot at x={target_x} km, found {selected.tolist()}")
    shot_index = int(selected[0])
    source_x = float(shots[shot_index, 0] * grid_scale_km)
    gather = born[shot_index]
    receiver_positions = np.asarray(experiment["receivers"], dtype=np.float64)[:, 0] * grid_scale_km
    time_positions = (np.arange(experiment["steps"], dtype=np.float64) + 1.0) * dt * time_unit_s
    print(f"selected_shot={shot_index} source_x_km={source_x:.16g}")
    print(f"born_gather_shape={gather.shape} gather_min={np.min(gather):.16e} gather_max={np.max(gather):.16e} gather_max_abs={np.max(np.abs(gather)):.16e}")

    x = np.arange(nx, dtype=np.float64) * grid_scale_km
    z = np.arange(nz, dtype=np.float64) * grid_scale_km
    extent = (float(x[0]), float(x[-1]), float(z[-1]), float(z[0]))
    cmap_signed = "RdBu_r"

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    background_image = axes[0, 0].imshow(background, extent=extent, aspect="auto", cmap="viridis", origin="upper")
    axes[0, 0].set_title("Smoothed Marmousi background")
    axes[0, 0].set_xlabel("Horizontal position (km)")
    axes[0, 0].set_ylabel("Depth (km)")
    mark_comparison_region(axes[0, 0])
    figure.colorbar(background_image, ax=axes[0, 0], label="Speed (km/s)")

    perturbation_limit = symmetric_limit(perturbation)
    perturbation_image = axes[0, 1].imshow(perturbation, extent=extent, aspect="auto", cmap=cmap_signed, vmin=-perturbation_limit, vmax=perturbation_limit, origin="upper")
    axes[0, 1].set_title("Short-wavelength perturbation")
    axes[0, 1].set_xlabel("Horizontal position (km)")
    axes[0, 1].set_ylabel("Depth (km)")
    mark_comparison_region(axes[0, 1])
    figure.colorbar(perturbation_image, ax=axes[0, 1], label="Velocity perturbation (km/s)")

    gather_limit = symmetric_limit(gather)
    gather_image = axes[1, 0].imshow(gather, extent=(float(receiver_positions[0]), float(receiver_positions[-1]), float(time_positions[-1]), float(time_positions[0])), aspect="auto", cmap=cmap_signed, vmin=-gather_limit, vmax=gather_limit, origin="upper")
    axes[1, 0].set_title(f"Born gather; source x = {source_x:.1f} km")
    axes[1, 0].set_xlabel("Receiver position (km)")
    axes[1, 0].set_ylabel("Time (s)")
    figure.colorbar(gather_image, ax=axes[1, 0], label="Scattered pressure (arbitrary units)")

    image_limit = symmetric_limit(image)
    rtm_image = axes[1, 1].imshow(image, extent=extent, aspect="auto", cmap=cmap_signed, vmin=-image_limit, vmax=image_limit, origin="upper")
    axes[1, 1].set_title("Checkpointed migration image")
    axes[1, 1].set_xlabel("Horizontal position (km)")
    axes[1, 1].set_ylabel("Depth (km)")
    mark_comparison_region(axes[1, 1])
    axes[1, 1].axhline(3.0, color="white", linewidth=0.8, alpha=0.8)
    figure.colorbar(rtm_image, ax=axes[1, 1], label="Adjoint image (arbitrary units)")

    output = ROOT / "artifacts/marmousi.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    output_size = output.stat().st_size
    assert output_size < 5 * 1024 * 1024
    print("qualitative_expectation=background smooth; perturbation curved/dipping; raw image upper-layer bands with weaker deeper amplitudes")
    print(f"Wrote artifacts/marmousi.png dimensions=2100x1500 bytes={output_size}")


if __name__ == "__main__":
    main()
