"""Block-bootstrap the Part 2 susceptibility peak fits."""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ARTIFACTS, WINDOW_NAMES, check_run


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
BLOCK_LENGTHS = (2000, 4000, 8000)
REPLICATES = 500
SEED = 20260916
TEMPERATURE_KEYS = tuple(range(2000, 2601, 50))
REFERENCE_TC = 2.26919


def read_window(lattice_size: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    directory = ARTIFACTS / WINDOW_NAMES[lattice_size]
    run = check_run(directory, lattice_size, WINDOW_NAMES[lattice_size])
    if run["measure"] != 100000 or len(run["t_grid"]) != 13:
        raise ValueError(f"unexpected window parameters in {directory}")
    values: dict[int, list[tuple[float, float]]] = {}
    with (directory / "series.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = round(row["T"] * 1000)
            values.setdefault(key, []).append((row["M"], row["M"] ** 2))
    if tuple(sorted(values)) != TEMPERATURE_KEYS or any(len(rows) != 100000 for rows in values.values()):
        raise ValueError(f"expected 13 temperatures with 100000 rows in {directory}")
    temperatures = np.array([values[key][0][0] * 0 + key / 1000 for key in TEMPERATURE_KEYS])
    # Store abs(M) and M^2 as two columns; all source rows are read only once.
    series = {
        key: np.array([(abs(m), m2) for m, m2 in values[key]], dtype=float)
        for key in TEMPERATURE_KEYS
    }
    return temperatures, series


def chi_from_columns(columns: np.ndarray, lattice_size: int, temperature: float) -> float:
    abs_m = columns[:, 0].mean()
    mean_m2 = columns[:, 1].mean()
    return lattice_size * lattice_size * (mean_m2 - abs_m**2) / temperature


def original_chi(series: dict[int, np.ndarray], lattice_size: int) -> np.ndarray:
    return np.array(
        [chi_from_columns(series[key], lattice_size, key / 1000) for key in TEMPERATURE_KEYS]
    )


def peak_fit(temperatures: np.ndarray, chi: np.ndarray) -> tuple[float | None, str | None, tuple[float, float] | None, np.ndarray | None]:
    maximum = int(np.argmax(chi))
    if maximum < 2 or maximum + 2 >= len(chi):
        return None, "incomplete five-point window", None, None
    fit_temperatures = temperatures[maximum - 2 : maximum + 3]
    coefficients = np.polyfit(fit_temperatures, chi[maximum - 2 : maximum + 3], 2)
    a, b, _ = coefficients
    if a >= 0:
        return None, "parabola does not open downward", None, None
    peak = -b / (2 * a)
    if not fit_temperatures.min() <= peak <= fit_temperatures.max():
        return None, "vertex outside five-point range", None, None
    return peak, None, (fit_temperatures[0], fit_temperatures[-1]), coefficients


def circular_moving_block(columns: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Resample ceil(N/B) circular consecutive blocks, then truncate to N rows."""
    sample_count = len(columns)
    block_count = (sample_count + block_length - 1) // block_length
    starts = rng.integers(0, sample_count, size=block_count)
    offsets = np.arange(block_length)
    indices = (starts[:, None] + offsets[None, :]) % sample_count
    return columns[indices.reshape(-1)[:sample_count]]


def bootstrap(
    data: dict[int, tuple[np.ndarray, dict[int, np.ndarray]]], rng: np.random.Generator
) -> tuple[dict[int, dict], dict[int, dict[int, list[tuple[np.ndarray, tuple[float, float]]]]]]:
    reports: dict[int, dict] = {}
    envelopes: dict[int, dict[int, list[tuple[np.ndarray, tuple[float, float]]]]] = {}
    temperatures = np.array([key / 1000 for key in TEMPERATURE_KEYS])
    for block_length in BLOCK_LENGTHS:
        reports[block_length] = {
            "attempts": REPLICATES,
            "valid_by_size": {},
            "failures_by_size": {},
            "peak_values": {32: [], 64: []},
            "tc_values": [],
            "tc_failures": Counter(),
        }
        envelopes[block_length] = {32: [], 64: []}

        valid_counts = {32: 0, 64: 0}
        failure_counts = {32: Counter(), 64: Counter()}
        for _ in range(REPLICATES):
            peaks = {}
            for lattice_size in (32, 64):
                chi = []
                for key in TEMPERATURE_KEYS:
                    source = data[lattice_size][1][key]
                    resampled = circular_moving_block(source, block_length, rng)
                    chi.append(chi_from_columns(resampled, lattice_size, key / 1000))
                peak, reason, interval, coefficients = peak_fit(temperatures, np.array(chi))
                if reason:
                    failure_counts[lattice_size][reason] += 1
                else:
                    valid_counts[lattice_size] += 1
                    peaks[lattice_size] = peak
                    reports[block_length]["peak_values"][lattice_size].append(peak)
                    envelopes[block_length][lattice_size].append((coefficients, interval))
            if len(peaks) == 2:
                reports[block_length]["tc_values"].append(2 * peaks[64] - peaks[32])
            else:
                reports[block_length]["tc_failures"]["one or both size fits invalid"] += 1
        reports[block_length]["valid_by_size"] = valid_counts
        reports[block_length]["failures_by_size"] = dict(failure_counts[32]), dict(failure_counts[64])
    return reports, envelopes


def write_report(
    data: dict[int, tuple[np.ndarray, dict[int, np.ndarray]]],
    reports: dict[int, dict],
) -> None:
    temperatures = np.array([key / 1000 for key in TEMPERATURE_KEYS])
    original_peaks = {}
    lines = [
        "Week 3 Part 3 block bootstrap report",
        f"method=circular moving-block bootstrap; each block is a consecutive ordered segment sampled from a uniform start",
        f"seed={SEED}, replicates={REPLICATES}, critical_window=2.0..2.6, temperatures=13",
        "tail_rule=circular moving-block bootstrap; sample ceil(N/B) blocks from uniform starts, wrap at the end, concatenate, then truncate to exactly N=100000",
        "",
    ]
    for lattice_size in (32, 64):
        chi = original_chi(data[lattice_size][1], lattice_size)
        peak, reason, _, _ = peak_fit(temperatures, chi)
        if reason:
            raise ValueError(f"original L={lattice_size} fit failed: {reason}")
        original_peaks[lattice_size] = peak
    original_tc = 2 * original_peaks[64] - original_peaks[32]
    lines.append(f"original_center_L32_T_peak={original_peaks[32]:.8f}")
    lines.append(f"original_center_L64_T_peak={original_peaks[64]:.8f}")
    lines.append(f"original_center_Tc={original_tc:.8f}")
    lines.append("")
    tc_errors = []
    for block_length in BLOCK_LENGTHS:
        report = reports[block_length]
        tc_values = np.array(report["tc_values"])
        tc_se = np.std(tc_values, ddof=1)
        tc_errors.append(tc_se)
        lines.extend(
            [
                f"block_length={block_length}",
                f"attempts={report['attempts']}, valid_Tc={len(tc_values)}, failed_Tc={report['attempts'] - len(tc_values)}",
                f"L32_valid={report['valid_by_size'][32]}, L32_failed={sum(report['failures_by_size'][0].values())}, L32_failure_reasons={report['failures_by_size'][0]}",
                f"L64_valid={report['valid_by_size'][64]}, L64_failed={sum(report['failures_by_size'][1].values())}, L64_failure_reasons={report['failures_by_size'][1]}",
                f"Tc_bootstrap_mean={tc_values.mean():.8f}, Tc_bootstrap_se_ddof1={tc_se:.8f}",
                f"L32_peak_se_ddof1={np.std(report['peak_values'][32], ddof=1):.8f}",
                f"L64_peak_se_ddof1={np.std(report['peak_values'][64], ddof=1):.8f}",
                f"circular_block_count={int(np.ceil(100000 / block_length))}, generated_rows={int(np.ceil(100000 / block_length)) * block_length}, truncated_rows={int(np.ceil(100000 / block_length)) * block_length - 100000}",
                "",
            ]
        )
    mean_error = float(np.mean(tc_errors))
    deviations = [abs(error - mean_error) / mean_error for error in tc_errors]
    stable = all(deviation <= 0.10 for deviation in deviations)
    lines.append(f"Tc_se_mean={mean_error:.8f}")
    lines.append("Tc_se_relative_deviations=" + ", ".join(f"{deviation:.6f}" for deviation in deviations))
    lines.append(f"Tc_sampling_error_stable={stable}")
    if not stable:
        lines.append("sampling error unresolved")
    if any(len(reports[length]["tc_values"]) < REPLICATES for length in BLOCK_LENGTHS):
        lines.append("Some Tc standard errors are based only on replicates with two valid size fits.")
    lines.append("This is sampling error only; it excludes finite-size extrapolation bias and five-point quadratic-fit bias.")
    (EVIDENCE / "bootstrap.txt").write_text("\n".join(lines) + "\n")


def plot(
    data: dict[int, tuple[np.ndarray, dict[int, np.ndarray]]],
    reports: dict[int, dict],
    envelopes: dict[int, dict[int, list[tuple[np.ndarray, tuple[float, float]]]]],
) -> None:
    temperatures = np.array([key / 1000 for key in TEMPERATURE_KEYS])
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    colors = {32: "tab:blue", 64: "tab:orange"}
    for axis, block_length in zip(axes, BLOCK_LENGTHS):
        for lattice_size in (32, 64):
            chi = original_chi(data[lattice_size][1], lattice_size)
            axis.plot(temperatures, chi, "o-", color=colors[lattice_size], label=f"L = {lattice_size} raw chi")
            peak, reason, interval, coefficients = peak_fit(temperatures, chi)
            if reason:
                continue
            fit_x = np.linspace(interval[0], interval[1], 100)
            axis.plot(fit_x, np.polyval(coefficients, fit_x), color=colors[lattice_size], linestyle="--", label=f"L = {lattice_size} raw fit")
            grid = np.linspace(interval[0], interval[1], 120)
            values = np.full((len(envelopes[block_length][lattice_size]), len(grid)), np.nan)
            for row, (bootstrap_coefficients, bootstrap_interval) in enumerate(envelopes[block_length][lattice_size]):
                mask = (grid >= bootstrap_interval[0]) & (grid <= bootstrap_interval[1])
                values[row, mask] = np.polyval(bootstrap_coefficients, grid[mask])
            lower = np.nanmin(values, axis=0)
            upper = np.nanmax(values, axis=0)
            axis.fill_between(grid, lower, upper, color=colors[lattice_size], alpha=0.12, label=f"L = {lattice_size} bootstrap fit envelope (not CI)")
        axis.axvline(REFERENCE_TC, color="black", linestyle=":", label="Tc = 2.26919")
        axis.set_title(f"Block length = {block_length}")
        axis.set_xlabel("Temperature T")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Susceptibility chi")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle("Block-bootstrap susceptibility peak fits\nShaded regions are envelopes of valid fitted parabolas, not confidence intervals")
    figure.tight_layout(rect=(0, 0.12, 1, 0.93))
    figure.savefig(EVIDENCE / "chi-bootstrap.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    data = {lattice_size: read_window(lattice_size) for lattice_size in (32, 64)}
    reports, envelopes = bootstrap(data, np.random.default_rng(SEED))
    write_report(data, reports)
    plot(data, reports, envelopes)
    print((EVIDENCE / "bootstrap.txt").read_text(), end="")
    print(f"saved {EVIDENCE / 'chi-bootstrap.png'}")


if __name__ == "__main__":
    main()
