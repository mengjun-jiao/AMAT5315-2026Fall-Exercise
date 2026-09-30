"""Plot and validate the official Week 5 reflector RTM evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


X_START, X_STOP = 7, 34
Z_START, Z_STOP = 10, 34
TRUE_REFLECTOR_Z = 21


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "inputs" / "reflector.json").open() as handle:
        experiment = json.load(handle)
    perturbation = np.asarray(experiment["perturbation"], dtype=np.float64)
    image = np.load(root / "artifacts" / "adjoint" / "image.npy")

    expected_shape = (experiment["nz"], experiment["nx"])
    if perturbation.shape != expected_shape:
        raise SystemExit(f"unexpected perturbation shape: {perturbation.shape}")
    if image.shape != expected_shape or image.dtype != np.float64:
        raise SystemExit(f"unexpected image array: shape={image.shape}, dtype={image.dtype}")
    if not np.isfinite(perturbation).all() or not np.isfinite(image).all():
        raise SystemExit("perturbation and image must contain only finite values")

    perturbation_window = perturbation[Z_START:Z_STOP, X_START:X_STOP]
    image_window = image[Z_START:Z_STOP, X_START:X_STOP]
    row_l2 = np.sqrt(np.sum(image_window * image_window, axis=1))
    peak_z = Z_START + int(np.argmax(row_l2))
    cell_difference = abs(peak_z - TRUE_REFLECTOR_Z)
    scale_km = experiment["dx"] * experiment["length_unit_m"] / 1000.0
    true_depth_km = TRUE_REFLECTOR_Z * scale_km
    peak_depth_km = peak_z * scale_km
    depth_difference_km = abs(peak_depth_km - true_depth_km)

    print(f"true_reflector_grid_z={TRUE_REFLECTOR_Z}")
    print(f"true_reflector_depth_km={true_depth_km:.12f}")
    print(f"detected_peak_grid_z={peak_z}")
    print(f"detected_peak_depth_km={peak_depth_km:.12f}")
    print(f"grid_cell_difference={cell_difference}")
    print(f"physical_depth_difference_km={depth_difference_km:.12f}")
    print(f"peak_row_l2={row_l2[peak_z - Z_START]:.15e}")

    if cell_difference > 1:
        raise SystemExit("reflector depth peak is outside the one-cell acceptance window")

    side_lobe_mask = np.ones(row_l2.shape, dtype=bool)
    side_lobe_mask[max(0, peak_z - Z_START - 1) : peak_z - Z_START + 2] = False
    if side_lobe_mask.any():
        side_lobe = float(np.max(row_l2[side_lobe_mask]))
        print(f"largest_shallow_side_lobe_row_l2={side_lobe:.15e}")
        print(f"side_lobe_to_peak_ratio={side_lobe / row_l2[peak_z - Z_START]:.15e}")

    x_indices = np.arange(X_START, X_STOP)
    z_indices = np.arange(Z_START, Z_STOP)
    x_km = x_indices * scale_km
    z_km = z_indices * scale_km
    half_cell = scale_km / 2.0
    extent = (
        x_km[0] - half_cell,
        x_km[-1] + half_cell,
        z_km[-1] + half_cell,
        z_km[0] - half_cell,
    )

    perturbation_limit = float(np.max(np.abs(perturbation_window)))
    image_limit = float(np.max(np.abs(image_window)))
    perturbation_norm = TwoSlopeNorm(vmin=-perturbation_limit, vcenter=0.0, vmax=perturbation_limit)
    image_norm = TwoSlopeNorm(vmin=-image_limit, vcenter=0.0, vmax=image_limit)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    field_axes = (axes[0], axes[1])
    field_titles = ("Known reflector", "Raw signed RTM image")
    field_data = (perturbation_window, image_window)
    field_norms = (perturbation_norm, image_norm)
    field_labels = ("Velocity change (km/s)", "Image (arbitrary units)")
    for axis, title, data, norm, label in zip(
        field_axes, field_titles, field_data, field_norms, field_labels
    ):
        image_artist = axis.imshow(data, extent=extent, origin="upper", aspect="auto", norm=norm, cmap="seismic")
        axis.axhline(true_depth_km, color="black", linestyle="--", linewidth=1.0, label="True depth")
        axis.set_title(title)
        axis.set_xlabel("Horizontal position (km)")
        axis.set_ylabel("Depth (km)")
        axis.legend(loc="lower right", fontsize=8)
        colorbar = figure.colorbar(image_artist, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label(label)

    profile_axis = axes[2]
    profile_axis.plot(row_l2, z_km, color="tab:blue", linewidth=2.0)
    profile_axis.scatter(
        row_l2[peak_z - Z_START],
        peak_depth_km,
        color="tab:red",
        zorder=3,
        label=f"Peak: z={peak_z}",
    )
    profile_axis.axhline(true_depth_km, color="black", linestyle="--", linewidth=1.0, label="True depth: z=21")
    profile_axis.set_title("Image depth profile")
    profile_axis.set_xlabel("Row L2 norm (arbitrary units)")
    profile_axis.set_ylabel("Depth (km)")
    profile_axis.set_ylim(z_km[-1] + half_cell, z_km[0] - half_cell)
    profile_axis.legend(loc="lower right", fontsize=8)

    output = root / "artifacts" / "adjoint" / "image.png"
    figure.savefig(output, dpi=180, format="png")
    plt.close(figure)
    print(f"image_png={output}")
    print(f"image_png_dimensions={plt.imread(output).shape[1]}x{plt.imread(output).shape[0]}")
    print(f"image_png_bytes={output.stat().st_size}")
    if output.stat().st_size >= 5_000_000:
        raise SystemExit("image.png exceeds the 5 MB limit")
    print("RTM image evidence validation: passed")


if __name__ == "__main__":
    main()
