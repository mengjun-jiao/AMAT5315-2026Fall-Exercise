"""Fit five-point quadratic susceptibility peaks for Week 3 Part 2."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import mean_abs_m, merged_data, susceptibility


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
REFERENCE_TC = 2.26919


def fit_peak(lattice_size: int) -> tuple[dict, np.ndarray, float]:
    data = merged_data(lattice_size)
    keys = sorted(key for key in data if 2000 <= key <= 2600)
    temperatures = np.array([data[key]["T"] for key in keys])
    values = np.array([susceptibility(data[key], lattice_size) for key in keys])
    peak_index = int(np.argmax(values))
    if peak_index < 2 or peak_index + 2 >= len(values):
        raise ValueError(f"L={lattice_size}: susceptibility peak is too close to the fitting-window boundary")
    selected_t = temperatures[peak_index - 2 : peak_index + 3]
    selected_chi = values[peak_index - 2 : peak_index + 3]
    coefficients = np.polyfit(selected_t, selected_chi, 2)
    a, b, _ = coefficients
    if a >= 0:
        raise ValueError(f"L={lattice_size}: quadratic fit does not open downward")
    fitted_peak = -b / (2 * a)
    if not selected_t.min() <= fitted_peak <= selected_t.max():
        raise ValueError(f"L={lattice_size}: fitted peak is outside the five-point temperature range")
    return data, selected_t, fitted_peak


def main() -> None:
    results = {}
    lines = []
    for lattice_size in (32, 64):
        data, fit_temperatures, fitted_peak = fit_peak(lattice_size)
        results[lattice_size] = fitted_peak
        lines.append(f"L={lattice_size} five-point fit:")
        for temperature in fit_temperatures:
            value = data[round(temperature * 1000)]
            lines.append(f"  T={temperature:.2f}, chi={susceptibility(value, lattice_size):.6f}")
        lines.append(f"  fitted T_peak={fitted_peak:.8f}")
    tc_estimate = 2 * results[64] - results[32]
    relative_error = abs(tc_estimate - REFERENCE_TC) / REFERENCE_TC
    low_temperature = []
    for lattice_size in (32, 64):
        value = merged_data(lattice_size)[1500]
        low_temperature.append(mean_abs_m(value))
        lines.append(f"L={lattice_size} T=1.5 mean_abs_M={low_temperature[-1]:.6f}")
    l64_t30 = merged_data(64)[3000]
    lines.append(f"L=64 T=3.0 mean_abs_M={mean_abs_m(l64_t30):.6f}")
    lines.extend(
        [
            f"Tc_est={tc_estimate:.8f}",
            f"reference_Tc={REFERENCE_TC:.5f}",
            f"relative_error={relative_error:.8f}",
            f"low_temperature_mean_abs_M_at_least_0.9={all(value >= 0.9 for value in low_temperature)}",
            f"critical_temperature_relative_error_below_2_percent={relative_error < 0.02}",
        ]
    )
    report = "\n".join(lines) + "\n"
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "peaks.txt").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
