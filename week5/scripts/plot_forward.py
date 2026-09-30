"""Plot common-scale forward shot gathers."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "artifacts" / "forward"
    with (output_dir / "run.json").open() as handle:
        run = json.load(handle)
    traces = np.load(output_dir / "traces.npy")
    experiment = run["experiment"]

    receivers = np.asarray(experiment["receivers"], dtype=np.float64)
    shots = np.asarray(experiment["shots"], dtype=np.float64)
    receiver_x = receivers[:, 0] * experiment["dx"] * experiment["length_unit_m"] / 1000.0
    time = (
        np.arange(traces.shape[1], dtype=np.float64) + 1.0
    ) * experiment["dt"] * experiment["time_unit_s"]
    vmax = float(np.max(np.abs(traces)))

    fig, axes = plt.subplots(1, traces.shape[0], figsize=(15, 5.5), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    image = None
    for shot_index, (axis, shot) in enumerate(zip(axes, shots)):
        image = axis.imshow(
            traces[shot_index],
            extent=(receiver_x.min(), receiver_x.max(), time[-1], time[0]),
            aspect="auto",
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        source_x = shot[0] * experiment["dx"] * experiment["length_unit_m"] / 1000.0
        axis.set_title(f"Shot {shot_index} (source x = {source_x:.2f} km)")
        axis.set_xlabel("Receiver position (km)")
        axis.set_xlim(receiver_x.min(), receiver_x.max())
        axis.set_ylim(time[-1], time[0])
    axes[0].set_ylabel("Time (s)")
    fig.colorbar(image, ax=axes.tolist(), label="Pressure", shrink=0.9)
    fig.suptitle("Forward pressure gathers", y=1.02)
    output_path = output_dir / "gathers.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"gathers_plot={output_path}")
    print(f"common_pressure_scale={vmax:.12f}")


if __name__ == "__main__":
    main()
