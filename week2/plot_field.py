"""Render Rust-generated energy and force values; keep CSV data in memory."""

import io
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def main():
    week_dir = Path(__file__).resolve().parent
    result = subprocess.run(
        [
            "cargo", "run", "--quiet", "--manifest-path",
            str(week_dir / "md" / "Cargo.toml"), "--example", "field",
        ],
        check=True, capture_output=True, text=True,
    )
    data = np.genfromtxt(io.StringIO(result.stdout), delimiter=",", names=True)
    ix, iy = data["ix"].astype(int), data["iy"].astype(int)
    energy = np.ma.masked_all((iy.max() + 1, ix.max() + 1))
    energy[iy, ix] = data["energy"]
    xs, ys = np.unique(data["x"]), np.unique(data["y"])

    fig, ax = plt.subplots(figsize=(8.5, 8))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    background = ax.pcolormesh(
        xs, ys, energy, shading="nearest", cmap=cmap, vmin=-1, vmax=1,
        rasterized=True,
    )
    fig.colorbar(background, ax=ax, extend="max", pad=0.03, shrink=0.8,
                 label="Potential energy U(r) [reduced units]")

    # Sample the Rust vectors, then multiply both components by the same
    # positive scale. The fourth root keeps weak, distant forces visible.
    arrows = data[(ix % 16 == 0) & (iy % 16 == 0)]
    magnitude = np.hypot(arrows["fx"], arrows["fy"])
    root_magnitude = magnitude ** 0.25
    length = 0.28 * root_magnitude / (1.0 + root_magnitude)
    scale = np.divide(length, magnitude, out=np.zeros_like(magnitude),
                      where=magnitude > 0)
    ax.quiver(
        arrows["x"], arrows["y"], arrows["fx"] * scale, arrows["fy"] * scale,
        angles="xy", scale_units="xy", scale=1, pivot="tail",
        color="#171717", width=0.0035,
    )
    r0 = 2 ** (1 / 6)
    ax.add_patch(Circle((0, 0), r0, fill=False, linestyle="--",
                        linewidth=1.8, color="black",
                        label=rf"$r_0=2^{{1/6}}\approx {r0:.3f}$"))
    ax.plot(0, 0, "ko", markersize=6, label="Fixed atom")
    ax.text(0, -0.35, "$r < 0.85$\nmasked", ha="center", va="center", fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.6, alpha=0.5)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="x [reduced units]",
           ylabel="y [reduced units]", title="Lennard-Jones potential and force on a second atom")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.text(
        0.5, 0.025,
        "Display: U > 1 uses the top color; r < 0.85 omitted.\n"
        "Arrow length = 0.28 |F|^(1/4) / (1 + |F|^(1/4)); directions unchanged.\n"
        "Energy and force data: Rust md::energy / md::force.",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    output = week_dir / "field.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
