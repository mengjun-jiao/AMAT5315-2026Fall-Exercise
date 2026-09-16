"""Create the Week 3 Part 2 magnetization and susceptibility plots."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import mean_abs_m, merged_data, susceptibility


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
TC = 2 / np.log(1 + np.sqrt(2))


def magnetization_plot() -> None:
    data = merged_data(64)
    temperatures = np.array([data[key]["T"] for key in sorted(data)])
    magnetizations = np.array([mean_abs_m(data[key]) for key in sorted(data)])
    # Use an independent dense grid so the analytic curve is not connected
    # across Tc through the sparse simulation temperatures.
    theory_below_tc = np.linspace(temperatures.min(), TC, 500, endpoint=False)
    theory_below_tc = np.append(theory_below_tc, TC)
    theory_values_below_tc = np.zeros_like(theory_below_tc)
    below_tc = theory_below_tc < TC
    theory_values_below_tc[below_tc] = (
        1 - np.sinh(2 / theory_below_tc[below_tc]) ** -4
    ) ** (1 / 8)
    theory_above_tc = np.linspace(TC, temperatures.max(), 250)
    theory_values_above_tc = np.zeros_like(theory_above_tc)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(temperatures, magnetizations, "o-", label="Metropolis, L = 64")
    axis.plot(
        theory_below_tc,
        theory_values_below_tc,
        "--",
        color="tab:orange",
        label="Infinite lattice theory",
    )
    axis.plot(theory_above_tc, theory_values_above_tc, "--", color="tab:orange")
    axis.axvline(TC, color="black", linestyle=":", label=f"Tc = {TC:.5f}")
    axis.set_xlabel("Temperature T")
    axis.set_ylabel("Mean absolute magnetization")
    axis.set_title("Ising magnetization")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(EVIDENCE / "magnetization.png", dpi=160)
    plt.close(figure)


def susceptibility_plot() -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for lattice_size, color in [(32, "tab:blue"), (64, "tab:orange")]:
        data = merged_data(lattice_size)
        window_keys = sorted(key for key in data if 2000 <= key <= 2600)
        temperatures = np.array([data[key]["T"] for key in window_keys])
        values = np.array([susceptibility(data[key], lattice_size) for key in window_keys])
        axis.plot(temperatures, values, "o-", color=color, label=f"L = {lattice_size}")
    axis.set_xlabel("Temperature T")
    axis.set_ylabel("Susceptibility chi")
    axis.set_title("Ising susceptibility in the critical window")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(EVIDENCE / "susceptibility.png", dpi=160)
    plt.close(figure)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    magnetization_plot()
    susceptibility_plot()
    print(f"saved {EVIDENCE / 'magnetization.png'}")
    print(f"saved {EVIDENCE / 'susceptibility.png'}")
    print(f"theoretical Tc = {TC:.8f}")


if __name__ == "__main__":
    main()
