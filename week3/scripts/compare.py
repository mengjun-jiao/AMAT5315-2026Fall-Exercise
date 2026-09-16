"""Compare Metropolis and Wolff window samples with block bootstrap errors."""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WEEK3 = Path(__file__).resolve().parents[1]
ARTIFACTS = WEEK3 / "artifacts"
EVIDENCE = WEEK3 / "evidence"
TEMPERATURE_KEYS = tuple(range(2000, 2601, 50))
TEMPERATURES = np.array([key / 1000 for key in TEMPERATURE_KEYS])
BLOCK_LENGTHS = (2000, 4000, 8000)
REPLICATES = 500
SEED = 20260916
REFERENCE_TC = 2.26919


def read_series(name: str, lattice_size: int, update: str) -> dict[int, np.ndarray]:
    directory = ARTIFACTS / name
    run = json.loads((directory / "run.json").read_text())
    expected_time = "cluster_flip" if update == "wolff" else "sweep"
    if run["L"] != lattice_size or run["update"] != update or run["time_unit"] != expected_time:
        raise ValueError(f"unexpected metadata in {directory}")
    if run["measure"] != 100000 or run["sample_every"] != 1:
        raise ValueError(f"unexpected measurement metadata in {directory}")
    rows = {key: np.empty((100000, 2), dtype=float) for key in TEMPERATURE_KEYS}
    counts = {key: 0 for key in TEMPERATURE_KEYS}
    with (directory / "series.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            key = round(row["T"] * 1000)
            if key not in rows or counts[key] == 100000:
                raise ValueError(f"unexpected temperature or row count in {directory}")
            rows[key][counts[key]] = (abs(row["M"]), row["M"] ** 2)
            counts[key] += 1
            if update == "wolff" and not 1 <= row.get("cluster_size", 0) <= lattice_size * lattice_size:
                raise ValueError(f"invalid cluster size in {directory}")
    if any(count != 100000 for count in counts.values()):
        raise ValueError(f"expected 13 temperatures and 100000 rows in {directory}")
    return rows


class CircularMovingBlocks:
    """Fast circular moving-block resampling of two observable columns."""

    def __init__(self, columns: np.ndarray, block_length: int):
        self.columns = columns
        self.block_length = block_length
        self.block_count = (len(columns) + block_length - 1) // block_length

    def sample_columns(self, rng: np.random.Generator) -> np.ndarray:
        starts = rng.integers(0, len(self.columns), size=self.block_count)
        offsets = np.arange(self.block_length)
        indices = (starts[:, None] + offsets[None, :]) % len(self.columns)
        return self.columns[indices.reshape(-1)[: len(self.columns)]]

    def sample_stats(self, rng: np.random.Generator) -> tuple[float, float]:
        sampled = self.sample_columns(rng)
        return float(sampled[:, 0].mean()), float(sampled[:, 1].mean())


def chi(mean_abs: float, mean_m2: float, lattice_size: int, temperature: float) -> float:
    return lattice_size * lattice_size * (mean_m2 - mean_abs**2) / temperature


def peak_fit(values: np.ndarray) -> tuple[float | None, str | None, tuple[float, float] | None, np.ndarray | None]:
    index = int(np.argmax(values))
    if index < 2 or index + 2 >= len(values):
        return None, "incomplete five-point window", None, None
    fit_t = TEMPERATURES[index - 2 : index + 3]
    coefficients = np.polyfit(fit_t, values[index - 2 : index + 3], 2)
    a, b, _ = coefficients
    if a >= 0:
        return None, "parabola does not open downward", None, None
    peak = -b / (2 * a)
    if not fit_t.min() <= peak <= fit_t.max():
        return None, "vertex outside five-point range", None, None
    return peak, None, (fit_t[0], fit_t[-1]), coefficients


def original_chi(series: dict[int, np.ndarray], lattice_size: int) -> np.ndarray:
    return np.array([chi(value[:, 0].mean(), value[:, 1].mean(), lattice_size, key / 1000) for key, value in sorted(series.items())])


def stream_seed(algorithm_index: int, lattice_index: int, block_index: int) -> int:
    return SEED + 10000 * algorithm_index + 100 * lattice_index + block_index


def bootstrap_algorithm(
    series: dict[int, np.ndarray], lattice_size: int, algorithm_index: int, lattice_index: int
) -> dict[int, dict]:
    reports = {}
    for block_index, block_length in enumerate(BLOCK_LENGTHS):
        rng = np.random.default_rng(stream_seed(algorithm_index, lattice_index, block_index))
        blocks = {key: CircularMovingBlocks(value, block_length) for key, value in series.items()}
        means = []
        chis = []
        fits = []
        failures = Counter()
        for _ in range(REPLICATES):
            mean_values = []
            chi_values = []
            for key in TEMPERATURE_KEYS:
                mean_abs, mean_m2 = blocks[key].sample_stats(rng)
                mean_values.append(mean_abs)
                chi_values.append(chi(mean_abs, mean_m2, lattice_size, key / 1000))
            peak, reason, interval, coefficients = peak_fit(np.array(chi_values))
            if reason:
                failures[reason] += 1
            else:
                means.append(mean_values)
                chis.append(chi_values)
                fits.append((peak, interval, coefficients))
        reports[block_length] = {
            "seed": stream_seed(algorithm_index, lattice_index, block_index),
            "mean_values": np.asarray(means),
            "chi_values": np.asarray(chis),
            "fits": fits,
            "valid": len(fits),
            "failures": failures,
        }
    return reports


def write_report(data: dict, results: dict) -> None:
    lines = [
        "Week 3 Part 4 sampler agreement and Wolff critical-temperature report",
        "method=circular moving-block bootstrap; each block is a consecutive ordered segment from a uniform start, wrapping at the end",
        f"root_seed={SEED}, replicates={REPLICATES}, block_lengths={BLOCK_LENGTHS}",
        "tail_rule=sample ceil(N/B) blocks and truncate the concatenation to exactly N=100000; no independent tail shuffling",
        "",
    ]
    original = {}
    for algorithm in ("metropolis", "wolff"):
        for lattice_size in (32, 64):
            values = original_chi(data[algorithm][lattice_size], lattice_size)
            peak, reason, interval, coefficients = peak_fit(values)
            if reason:
                raise ValueError(f"original {algorithm} L={lattice_size} fit failed: {reason}")
            original[(algorithm, lattice_size)] = (peak, interval, coefficients)
    original_tc = 2 * original[("wolff", 64)][0] - original[("wolff", 32)][0]
    wolff_l32_peak = original[("wolff", 32)][0]
    wolff_l64_peak = original[("wolff", 64)][0]
    lines += [
        f"wolff_original_L32_T_peak={wolff_l32_peak:.8f}",
        f"wolff_original_L64_T_peak={wolff_l64_peak:.8f}",
        f"wolff_original_Tc={original_tc:.8f}",
        f"wolff_original_relative_error={abs(original_tc - REFERENCE_TC) / REFERENCE_TC:.8f}",
        "",
        "L=64 sampler agreement by temperature (mean_abs_M and bootstrap SE for block lengths 2000, 4000, 8000):",
    ]
    agreement = {}
    for key, temperature in zip(TEMPERATURE_KEYS, TEMPERATURES):
        means = {}
        errors = {}
        stable = {}
        for algorithm in ("metropolis", "wolff"):
            means[algorithm] = float(data[algorithm][64][key][:, 0].mean())
            errors[algorithm] = np.array([np.std(results[algorithm][64][length]["mean_values"][:, key // 50 - 40], ddof=1) for length in BLOCK_LENGTHS])
            average = errors[algorithm].mean()
            stable[algorithm] = bool(np.all(np.abs(errors[algorithm] - average) / average <= 0.10))
        d_values = np.abs(means["metropolis"] - means["wolff"]) / np.sqrt(errors["metropolis"] ** 2 + errors["wolff"] ** 2)
        agreement[temperature] = (means, errors, stable, d_values)
        lines.append(
            f"T={temperature:.2f} M_metropolis={means['metropolis']:.6f} "
            f"SE_metropolis=" + ",".join(f"{value:.8f}" for value in errors["metropolis"]) +
            f" stable_metropolis={stable['metropolis']} M_wolff={means['wolff']:.6f} "
            f"SE_wolff=" + ",".join(f"{value:.8f}" for value in errors["wolff"]) +
            f" stable_wolff={stable['wolff']} d=" + ",".join(f"{value:.4f}" for value in d_values)
        )
    lines += ["", "T=2.3 agreement interpretation:"]
    d_values = agreement[2.3][3]
    lines.append("d_by_block_length=" + ", ".join(f"{value:.6f}" for value in d_values))
    both_stable = all(agreement[2.3][2].values())
    lines.append(f"both_error_sets_stable={both_stable}")
    if both_stable and np.all(d_values <= 3):
        lines.append("agreement")
    elif np.any(d_values > 3):
        lines.append("discrepancy needing investigation")
    else:
        lines.append("agreement provisional")
    lines.append("Part 3 binning-platform conclusion remains: sampling error unresolved for the Metropolis critical series.")
    lines += ["", "Wolff Tc bootstrap:"]
    tc_errors = []
    for block_length in BLOCK_LENGTHS:
        l32 = results["wolff"][32][block_length]
        l64 = results["wolff"][64][block_length]
        valid = min(l32["valid"], l64["valid"])
        tc_values = np.array([2 * l64["fits"][i][0] - l32["fits"][i][0] for i in range(valid)])
        tc_se = np.std(tc_values, ddof=1)
        tc_errors.append(tc_se)
        lines += [
            f"block_length={block_length} seed_L32={l32['seed']} seed_L64={l64['seed']}",
            f"attempts=500 valid_L32={l32['valid']} valid_L64={l64['valid']} valid_Tc={valid} failed_Tc={500 - valid}",
            f"L32_failure_reasons={dict(l32['failures'])} L64_failure_reasons={dict(l64['failures'])}",
            f"L32_peak_SE={np.std([fit[0] for fit in l32['fits']], ddof=1):.8f} L64_peak_SE={np.std([fit[0] for fit in l64['fits']], ddof=1):.8f} Tc_SE={tc_se:.8f}",
        ]
    mean_error = float(np.mean(tc_errors))
    deviations = [abs(value - mean_error) / mean_error for value in tc_errors]
    lines += [
        f"Tc_SE_mean={mean_error:.8f}",
        "Tc_SE_relative_deviations=" + ", ".join(f"{value:.6f}" for value in deviations),
        f"Tc_sampling_error_stable={all(value <= 0.10 for value in deviations)}",
        "This is sampling error only; it excludes finite-size extrapolation bias and five-point quadratic-fit bias.",
    ]
    (EVIDENCE / "sampler-compare.txt").write_text("\n".join(lines) + "\n")


def plot(data: dict, results: dict) -> None:
    figure, (mag_axis, chi_axis) = plt.subplots(1, 2, figsize=(14, 5))
    for algorithm, color in (("metropolis", "tab:blue"), ("wolff", "tab:orange")):
        means = np.array([data[algorithm][64][key][:, 0].mean() for key in TEMPERATURE_KEYS])
        errors = results[algorithm][64][4000]["mean_values"].std(axis=0, ddof=1)
        mag_axis.errorbar(TEMPERATURES, means, yerr=errors, fmt="o-", color=color, capsize=2, label=f"{algorithm} (B=4000)")
    mag_axis.set_xlabel("Temperature T")
    mag_axis.set_ylabel("Mean absolute magnetization")
    mag_axis.set_title("L = 64 sampler agreement (provisional errors)")
    mag_axis.grid(alpha=0.25)
    mag_axis.legend()

    for lattice_size, color in ((32, "tab:blue"), (64, "tab:orange")):
        values = original_chi(data["wolff"][lattice_size], lattice_size)
        chi_axis.plot(TEMPERATURES, values, "o-", color=color, label=f"L = {lattice_size} raw chi")
        peak, interval, coefficients = (lambda fit: (fit[0], fit[2], fit[3]))(peak_fit(values))
        fit_x = np.linspace(interval[0], interval[1], 100)
        chi_axis.plot(fit_x, np.polyval(coefficients, fit_x), "--", color=color, label=f"L = {lattice_size} five-point fit")
        chi_axis.axvline(peak, color=color, linestyle=":", alpha=0.7)
    wolff_peaks = [peak_fit(original_chi(data["wolff"][size], size))[0] for size in (32, 64)]
    wolff_tc = 2 * wolff_peaks[1] - wolff_peaks[0]
    tc_errors = []
    for block_length in BLOCK_LENGTHS:
        l32 = results["wolff"][32][block_length]["fits"]
        l64 = results["wolff"][64][block_length]["fits"]
        count = min(len(l32), len(l64))
        tc_errors.append(np.std([2 * l64[i][0] - l32[i][0] for i in range(count)], ddof=1))
    chi_axis.axvline(REFERENCE_TC, color="black", linestyle=":", label="Tc = 2.26919")
    chi_axis.axvline(wolff_tc, color="tab:green", linestyle="--", label=f"Wolff extrapolated Tc = {wolff_tc:.5f} (SE B=4000: {tc_errors[1]:.5f})")
    chi_axis.set_xlabel("Temperature T")
    chi_axis.set_ylabel("Susceptibility chi")
    chi_axis.set_title("Wolff susceptibility and peak fits")
    chi_axis.grid(alpha=0.25)
    chi_axis.legend(fontsize=8)
    figure.suptitle("Sampler comparison and Wolff critical temperature\nError bars use B=4000; bootstrap spread is not a 95% confidence interval")
    figure.tight_layout()
    figure.savefig(EVIDENCE / "magnetization-compare.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    data = {
        "metropolis": {32: read_series("window-l32", 32, "metropolis"), 64: read_series("window-l64", 64, "metropolis")},
        "wolff": {32: read_series("wolff-l32", 32, "wolff"), 64: read_series("wolff-l64", 64, "wolff")},
    }
    results = {"metropolis": {}, "wolff": {}}
    for algorithm_index, algorithm in enumerate(("metropolis", "wolff")):
        for lattice_index, lattice_size in enumerate((32, 64)):
            results[algorithm][lattice_size] = bootstrap_algorithm(data[algorithm][lattice_size], lattice_size, algorithm_index, lattice_index)
    write_report(data, results)
    plot(data, results)
    print((EVIDENCE / "sampler-compare.txt").read_text(), end="")
    print(f"saved {EVIDENCE / 'magnetization-compare.png'}")


if __name__ == "__main__":
    main()
