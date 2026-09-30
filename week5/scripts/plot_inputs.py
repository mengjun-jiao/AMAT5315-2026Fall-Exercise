"""Plot the official reflector geometry and source pulse."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ricker(frequency: float, peak_time: float, time: np.ndarray) -> np.ndarray:
    theta = np.pi * frequency * (time - peak_time)
    return (1.0 - 2.0 * theta**2) * np.exp(-theta**2)


def local_minima(values: np.ndarray) -> np.ndarray:
    return np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])) + 1


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "inputs" / "reflector.json"
    output_path = root / "artifacts" / "inputs.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as handle:
        experiment = json.load(handle)

    nx, nz = experiment["nx"], experiment["nz"]
    dx = experiment["dx"]
    length_unit_m = experiment["length_unit_m"]
    time_unit_s = experiment["time_unit_s"]
    x_scale_km = dx * length_unit_m / 1000.0
    z_scale_km = x_scale_km
    width_km = (nx - 1) * x_scale_km
    depth_km = (nz - 1) * z_scale_km
    x = np.arange(nx) * x_scale_km
    z = np.arange(nz) * z_scale_km
    xx, zz = np.meshgrid(x, z)
    background = np.asarray(experiment["background"], dtype=np.float64)
    perturbation = np.asarray(experiment["perturbation"], dtype=np.float64)

    reduced_time = np.arange(experiment["steps"], dtype=np.float64) * experiment["dt"]
    physical_time = reduced_time * time_unit_s
    pulse = ricker(
        experiment["source_frequency"],
        experiment["source_peak_time"],
        reduced_time,
    )
    minima = local_minima(pulse)
    troughs = minima[np.argsort(pulse[minima])[:2]]
    troughs = troughs[np.argsort(physical_time[troughs])]
    peak_index = int(np.argmax(pulse))
    physical_frequency_hz = experiment["source_frequency"] / time_unit_s

    fig, (geometry_ax, pulse_ax) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    geometry_ax.imshow(
        background,
        extent=(0.0, width_km, depth_km, 0.0),
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )
    geometry_ax.contour(
        xx,
        zz,
        perturbation,
        levels=[0.5 * float(np.nanmax(perturbation))],
        colors="white",
        linewidths=1.5,
    )
    shots = np.asarray(experiment["shots"], dtype=np.float64)
    receivers = np.asarray(experiment["receivers"], dtype=np.float64)
    geometry_ax.scatter(
        shots[:, 0] * x_scale_km,
        shots[:, 1] * z_scale_km,
        marker="*",
        s=100,
        color="tab:red",
        edgecolor="black",
        label="Shots",
        zorder=4,
    )
    geometry_ax.scatter(
        receivers[:, 0] * x_scale_km,
        receivers[:, 1] * z_scale_km,
        marker="v",
        s=35,
        color="tab:cyan",
        edgecolor="black",
        label="Receivers",
        zorder=4,
    )
    sponge_width = experiment["sponge_width"] * x_scale_km
    geometry_ax.axvline(sponge_width, color="white", linestyle="--", linewidth=0.9, label="Sponge inner edge")
    geometry_ax.axvline(width_km - sponge_width, color="white", linestyle="--", linewidth=0.9)
    geometry_ax.axhline(sponge_width, color="white", linestyle="--", linewidth=0.9)
    geometry_ax.axhline(depth_km - sponge_width, color="white", linestyle="--", linewidth=0.9)
    geometry_ax.set_title("Reflector acquisition geometry")
    geometry_ax.set_xlabel("Horizontal position (km)")
    geometry_ax.set_ylabel("Depth (km)")
    geometry_ax.set_xlim(0.0, width_km)
    geometry_ax.set_ylim(depth_km, 0.0)
    geometry_ax.legend(loc="lower right")
    geometry_ax.text(
        0.02,
        0.04,
        f"Background: {float(np.max(background)):.2f} km/s\n"
        f"Reflector max: +{float(np.max(perturbation)):.2f} km/s",
        transform=geometry_ax.transAxes,
        color="white",
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 3},
    )

    pulse_ax.plot(physical_time, pulse, color="tab:blue", linewidth=1.8)
    pulse_ax.axhline(0.0, color="black", linewidth=0.7)
    pulse_ax.scatter(physical_time[peak_index], pulse[peak_index], color="tab:red", zorder=3)
    pulse_ax.set_title("Ricker source pulse")
    pulse_ax.set_xlabel("Time (s)")
    pulse_ax.set_ylabel("Source pulse")
    pulse_ax.grid(alpha=0.25)

    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"source_peak_value={pulse[peak_index]:.12f}")
    print(f"source_peak_time_s={physical_time[peak_index]:.12f}")
    print(f"source_peak_frequency_hz={physical_frequency_hz:.12f}")
    for index in troughs:
        print(f"trough_time_s={physical_time[index]:.12f} trough_value={pulse[index]:.12f}")
    print(f"inputs_plot={output_path}")


if __name__ == "__main__":
    main()
