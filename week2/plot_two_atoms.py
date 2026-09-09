import io
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    week = Path(__file__).resolve().parent
    result = subprocess.run([
        "cargo", "run", "--quiet", "--manifest-path", str(week / "md/Cargo.toml"),
        "--example", "two_atoms",
    ], check=True, capture_output=True, text=True)
    data = np.genfromtxt(io.StringIO(result.stdout), delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    runs = {"Euler": 500, "VelocityVerlet": 500, "VelocityVerletLong": 5000}
    assert set(data["method"]) == set(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    initial_energies = []
    for method, steps in runs.items():
        rows = data[data["method"] == method]
        assert len(rows) == steps + 1
        np.testing.assert_array_equal(rows["step"], np.arange(steps + 1))
        np.testing.assert_allclose(rows["time"], np.arange(steps + 1) * 0.01, atol=1e-12)
        for field in ("time", "kinetic", "potential", "total", "e0", "delta"):
            assert np.isfinite(rows[field]).all()
        np.testing.assert_allclose(rows["total"], rows["kinetic"] + rows["potential"])
        np.testing.assert_allclose(rows["e0"], rows["total"][0], rtol=0, atol=1e-12)
        initial_energies.append(rows["e0"][0])
        col = 1 if steps == 5000 else 0
        label = "VelocityVerlet" if col == 1 else method
        displayed = rows["delta"] * (1000 if col == 1 else 1)
        axes[col].plot(rows["time"], displayed, label=label)
        print(f"{method}: max(abs(delta))={max(abs(rows['delta'])):.8g}, "
              f"final delta={rows['delta'][-1]:.8g}")
    np.testing.assert_allclose(initial_energies, initial_energies[0], rtol=0, atol=0)
    short = data[data["method"] == "VelocityVerlet"]
    long = data[data["method"] == "VelocityVerletLong"]
    for field in ("time", "kinetic", "potential", "total", "e0", "delta"):
        np.testing.assert_allclose(short[field], long[field][:501], rtol=0, atol=1e-12)
    for col, title in enumerate(("500 steps: Euler and Verlet", "5000 steps: Verlet observation")):
        ax = axes[col]
        ax.set_title(title)
        ax.set_ylabel("Relative energy error ×1000" if col == 1 else "delta = (E - E0) / |E0|")
        ax.set_xlabel("Time [reduced units]")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Two atoms: (0,0), (1.2,0); v=0; mass=1; dt=0.01; open boundaries")
    fig.tight_layout()
    fig.savefig(week / "dimer.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
