"""Autocorrelation and binning analysis for the Week 3 Part 3 data."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ARTIFACTS, COARSE_NAMES, WINDOW_NAMES, check_run, mean_abs_m, merged_data


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"
WINDOW_KEYS = range(2000, 2601, 50)


def series_by_temperature(path: Path) -> dict[int, np.ndarray]:
    values: dict[int, list[float]] = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = round(row["T"] * 1000)
            values.setdefault(key, []).append(abs(row["M"]))
    return {key: np.asarray(value, dtype=float) for key, value in values.items()}


def selected_series(lattice_size: int) -> dict[int, np.ndarray]:
    coarse_dir = ARTIFACTS / COARSE_NAMES[lattice_size]
    window_dir = ARTIFACTS / WINDOW_NAMES[lattice_size]
    check_run(coarse_dir, lattice_size, COARSE_NAMES[lattice_size])
    check_run(window_dir, lattice_size, WINDOW_NAMES[lattice_size])
    coarse = series_by_temperature(coarse_dir / "series.jsonl")
    window = series_by_temperature(window_dir / "series.jsonl")
    selected = {key: value for key, value in coarse.items() if key not in window}
    selected.update(window)
    if len(selected) != 27:
        raise ValueError(f"expected 27 selected temperatures for L={lattice_size}")
    return selected


def autocorrelation(series: np.ndarray) -> np.ndarray:
    centered = series - series.mean()
    size = len(centered)
    fft_size = 1 << (2 * size - 1).bit_length()
    spectrum = np.fft.rfft(centered, fft_size)
    covariance = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)[:size]
    if covariance[0] <= 0:
        raise ValueError("constant series has no defined autocorrelation")
    return covariance / covariance[0]


def tau_with_cutoff(rho: np.ndarray) -> tuple[float, int, bool]:
    tau = 0.5
    for lag in range(1, len(rho)):
        tau += rho[lag]
        if lag > 6 * tau:
            return tau, lag, True
    return tau, len(rho) - 1, False


def standard_error(series: np.ndarray) -> float:
    return float(np.std(series, ddof=1) / np.sqrt(len(series)))


def block_error(series: np.ndarray, block_length: int) -> tuple[float, int, int]:
    blocks = len(series) // block_length
    discarded = len(series) % block_length
    if blocks < 2:
        return float("nan"), blocks, discarded
    means = series[: blocks * block_length].reshape(blocks, block_length).mean(axis=1)
    return standard_error(means), blocks, discarded


def analyze(series: np.ndarray) -> dict[str, float | int | bool]:
    rho = autocorrelation(series)
    tau, cutoff, resolved = tau_with_cutoff(rho)
    naive = standard_error(series)
    blocked, blocks, discarded = block_error(series, len(series) // 50)
    return {
        "n": len(series),
        "mean": float(series.mean()),
        "naive": naive,
        "blocked": blocked,
        "ratio": blocked / naive,
        "tau": tau,
        "cutoff": cutoff,
        "resolved": resolved,
        "blocks": blocks,
        "discarded": discarded,
    }


def write_errors(all_series: dict[int, dict[int, np.ndarray]]) -> dict[tuple[int, int], dict]:
    report = [
        "L T n mean_abs_M naive_se block50_se ratio tau_int cutoff_lag cutoff_status",
    ]
    results = {}
    for lattice_size in (32, 64):
        for key in sorted(all_series[lattice_size]):
            result = analyze(all_series[lattice_size][key])
            results[(lattice_size, key)] = result
            status = "resolved" if result["resolved"] else "sampling_error_unresolved"
            report.append(
                f"{lattice_size} {key / 1000:.2f} {result['n']} {result['mean']:.6f} "
                f"{result['naive']:.8f} {result['blocked']:.8f} {result['ratio']:.4f} "
                f"{result['tau']:.2f} {result['cutoff']} {status}"
            )
    report.append("")
    report.append("Binning details for L=64, T=2.3 (block_length blocks discarded_tail standard_error):")
    series = all_series[64][2300]
    for block_length in binning_lengths(len(series)):
        error, blocks, discarded = block_error(series, block_length)
        value = f"{error:.8f}" if np.isfinite(error) else "not_computed_fewer_than_two_blocks"
        report.append(f"{block_length} {blocks} {discarded} {value}")
    report.append("")
    report.append("Interpretation:")
    report.append("The 50-block standard error is provisional; fifty blocks alone do not establish a plateau.")
    report.append("The binning curve continues to vary at long block lengths and has too few blocks for a stable comparison.")
    report.append("sampling error unresolved")
    report.append("")
    report.append("Requested summary statistics:")
    for temperature_key in (1500, 2300, 3500):
        result = results[(64, temperature_key)]
        report.append(
            f"L=64 T={temperature_key / 1000:.1f}: n={result['n']}, "
            f"mean_abs_M={result['mean']:.6f}, naive_se={result['naive']:.8f}, "
            f"block50_se={result['blocked']:.8f}, ratio={result['ratio']:.4f}, "
            f"tau_int={result['tau']:.2f}, cutoff_lag={result['cutoff']}"
        )
    for lattice_size in (32, 64):
        peak_key = max(
            (key for key in all_series[lattice_size]),
            key=lambda key: results[(lattice_size, key)]["tau"],
        )
        report.append(
            f"L={lattice_size} maximum_tau_int={results[(lattice_size, peak_key)]['tau']:.2f} "
            f"at T={peak_key / 1000:.2f}"
        )
    tau_critical = results[(64, 2300)]["tau"]
    report.append(f"L=64 T=2.3 n_eff={100000 / (2 * tau_critical):.2f}")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "errors.txt").write_text("\n".join(report) + "\n")
    return results


def binning_lengths(size: int) -> list[int]:
    required = {1, 2000, 4000, 8000}
    lengths = set(required)
    lengths.update(int(value) for value in np.geomspace(1, size, 28))
    return sorted(value for value in lengths if value <= size)


def trace_plot(all_series: dict[int, dict[int, np.ndarray]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for axis, key, label in zip(axes, (2300, 3000), ("T = 2.3", "T = 3.0")):
        series = all_series[64][key][:2000]
        axis.plot(np.arange(1, len(series) + 1), series, linewidth=0.6)
        axis.set_ylabel("abs(M)")
        axis.set_title(f"L = 64, {label}")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Recorded sweep")
    figure.suptitle("Absolute magnetization traces")
    figure.tight_layout()
    figure.savefig(EVIDENCE / "trace.png", dpi=160)
    plt.close(figure)


def acf_binning_plot(all_series: dict[int, dict[int, np.ndarray]]) -> None:
    series = all_series[64][2300]
    rho = autocorrelation(series)
    tau, cutoff, _ = tau_with_cutoff(rho)
    lengths = binning_lengths(len(series))
    errors = []
    blocks = []
    for length in lengths:
        error, count, _ = block_error(series, length)
        errors.append(error)
        blocks.append(count)
    figure, (acf_axis, bin_axis) = plt.subplots(1, 2, figsize=(12, 4.8))
    acf_axis.plot(np.arange(min(5000, len(rho))), rho[: min(5000, len(rho))])
    acf_axis.axvline(cutoff, color="tab:red", linestyle="--", label=f"cutoff = {cutoff}")
    acf_axis.set_xlabel("Lag (sweeps)")
    acf_axis.set_ylabel("Autocorrelation of abs(M)")
    acf_axis.set_title(f"ACF, tau_int = {tau:.1f}")
    acf_axis.legend()
    acf_axis.grid(alpha=0.25)
    bin_axis.plot(lengths, errors, "o-")
    bin_axis.set_xscale("log")
    bin_axis.set_xlabel("Block length (sweeps)")
    bin_axis.set_ylabel("Standard error of mean abs(M)")
    bin_axis.set_title("Binning dependence")
    bin_axis.grid(alpha=0.25)
    for length, error, count in zip(lengths, errors, blocks):
        if np.isfinite(error):
            bin_axis.annotate(str(count), (length, error), fontsize=7, xytext=(0, 4), textcoords="offset points", ha="center")
    figure.tight_layout()
    figure.savefig(EVIDENCE / "acf-binning.png", dpi=160)
    plt.close(figure)


def tau_plot(all_series: dict[int, dict[int, np.ndarray]]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for lattice_size, color in ((32, "tab:blue"), (64, "tab:orange")):
        keys = sorted(all_series[lattice_size])
        taus = [tau_with_cutoff(autocorrelation(all_series[lattice_size][key]))[0] for key in keys]
        axis.plot([key / 1000 for key in keys], taus, "o-", color=color, label=f"L = {lattice_size}")
    tc = 2 / np.log(1 + np.sqrt(2))
    axis.axvline(tc, color="black", linestyle=":", label=f"Tc = {tc:.5f}")
    axis.set_yscale("log")
    axis.set_xlabel("Temperature T")
    axis.set_ylabel("Integrated autocorrelation time (sweeps)")
    axis.set_title("Metropolis autocorrelation time")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.tight_layout()
    figure.savefig(EVIDENCE / "tau.png", dpi=160)
    plt.close(figure)


def main() -> None:
    all_series = {lattice_size: selected_series(lattice_size) for lattice_size in (32, 64)}
    write_errors(all_series)
    trace_plot(all_series)
    acf_binning_plot(all_series)
    tau_plot(all_series)
    print("saved errors.txt, trace.png, acf-binning.png, and tau.png")


if __name__ == "__main__":
    main()
